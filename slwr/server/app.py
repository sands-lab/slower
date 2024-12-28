import asyncio
import sys
import concurrent.futures
from functools import partial
from logging import INFO, WARN
from typing import Optional, Tuple

import grpc
import grpc.aio

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.address import parse_address
from flwr.common import log
from flwr.server.history import History
from flwr.server.app import ServerConfig, run_fl
from flwr.proto.transport_pb2_grpc import add_FlowerServiceServicer_to_server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import valid_certificates

from slwr.server.grpc.flower_servicer import FlowerServiceServicer
from slwr.server.server_model.manager import ServerModelManager, GrpcServerModelManager
from slwr.server.strategy import Strategy
from slwr.server.grpc.servicer import ServerModelServicer
from slwr.server.server import Server
from slwr.common.utils import run_async
from slwr.proto import server_model_pb2_grpc


ADDRESS_DRIVER_API = "0.0.0.0:9091"
ADDRESS_FLEET_API_GRPC_RERE = "0.0.0.0:9092"
ADDRESS_FLEET_API_GRPC_BIDI = "[::]:8080"  # IPv6 to keep start_server compatible
ADDRESS_FLEET_API_REST = "0.0.0.0:9093"

DATABASE = ":flwr-in-memory-state:"


def init_defaults(
    server: Optional[Server],
    config: Optional[ServerConfig],
    strategy: Strategy,
    client_manager: Optional[ClientManager],
) -> Tuple[Server, ServerConfig]:
    """Create server instance if none was given."""

    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()

        server_model_manager = GrpcServerModelManager(
            init_server_model_fn=strategy.init_server_model_fn,
        )

        server = Server(
            client_manager=client_manager,
            strategy=strategy,
            server_model_manager=server_model_manager
        )
    elif strategy is not None:
        log(WARN, "Both server and strategy were provided, ignoring strategy")

    if config is None:
        config = ServerConfig()

    return server, config


async def start_grpc_server(  # pylint: disable=too-many-arguments
    client_manager: ClientManager,
    server_model_manager: ServerModelManager,
    server_address: str,
    max_concurrent_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    keepalive_time_ms: int = 210000,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> grpc.Server:

    options = [
        ("grpc.max_concurrent_streams", max(100, max_concurrent_workers)),
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
        ("grpc.keepalive_time_ms", keepalive_time_ms),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.keepalive_permit_without_calls", 0),
    ]

    server = grpc.aio.server(
        migration_thread_pool=concurrent.futures.ThreadPoolExecutor(max_concurrent_workers),
        maximum_concurrent_rpcs=max_concurrent_workers,
        options=options,
    )

    if certificates is not None:
        if not valid_certificates(certificates):
            sys.exit(1)

        root_certificate_b, certificate_b, private_key_b = certificates

        server_credentials = grpc.ssl_server_credentials(
            ((private_key_b, certificate_b),),
            root_certificates=root_certificate_b,
            require_client_auth=False,
        )
        server.add_secure_port(server_address, server_credentials)
    else:
        server.add_insecure_port(server_address)

    slwr_servicer = ServerModelServicer(server_model_manager)
    server_model_pb2_grpc.add_ServerModelServicer_to_server(slwr_servicer, server)

    flwr_servicer = FlowerServiceServicer(client_manager)
    add_FlowerServiceServicer_to_server(flwr_servicer, server)

    await server.start()

    return server


@run_async
async def start_server(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    server_address: str = ADDRESS_FLEET_API_GRPC_BIDI,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> History:

    event(EventType.START_SERVER_ENTER)

    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server IP address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Initialize server and server config
    initialized_server, initialized_config = init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )
    log(
        INFO,
        "Starting Flower server, config: %s",
        initialized_config,
    )

    # Start gRPC server
    grpc_server = await start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_model_manager=initialized_server.server_model_manager,
        server_address=address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    log(
        INFO,
        "Flower ECE: gRPC server running (%s rounds), SSL is %s",
        initialized_config.num_rounds,
        "enabled" if certificates is not None else "disabled",
    )

    # Start training
    try:
        run_fl_ = partial(run_fl, server=initialized_server, config=initialized_config)
        hist = await asyncio.to_thread(run_fl_)

    except Exception as e:  # pylint: disable=broad-except
        log(WARN, "Exception in start_server: %s", e)
        hist = None

    finally:
        await grpc_server.stop(grace=1)

    event(EventType.START_SERVER_LEAVE)

    return hist
