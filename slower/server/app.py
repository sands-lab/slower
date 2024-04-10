# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server app."""


import sys
from logging import INFO
from typing import Optional, Tuple

import grpc
from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.address import parse_address
from flwr.common.logger import log
from flwr.server.history import History
from flwr.server.app import ServerConfig, run_fl
from flwr.server.fleet.grpc_bidi.flower_service_servicer import FlowerServiceServicer
from flwr.proto.transport_pb2_grpc import add_FlowerServiceServicer_to_server
from flwr.server.fleet.grpc_bidi.grpc_server import generic_create_grpc_server
from flwr.server.client_manager import ClientManager

from slower.server.server_model.manager.server_model_manager import ServerModelManager
from slower.server.strategy.base_strategy import SlStrategy
from slower.server.grpc.server_segment_servicer import ServerSegmentServicer
from slower.server.server import Server
from slower.server.common import init_defaults
from slower.proto import server_segment_pb2_grpc


ADDRESS_DRIVER_API = "0.0.0.0:9091"
ADDRESS_FLEET_API_GRPC_RERE = "0.0.0.0:9092"
ADDRESS_FLEET_API_GRPC_BIDI = "[::]:8080"  # IPv6 to keep start_server compatible
ADDRESS_FLEET_API_REST = "0.0.0.0:9093"

DATABASE = ":flwr-in-memory-state:"


def start_grpc_server(  # pylint: disable=too-many-arguments
    client_manager: ClientManager,
    server_model_manager: ServerModelManager,
    server_address: str,
    max_concurrent_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    keepalive_time_ms: int = 210000,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> grpc.Server:

    servicer = FlowerServiceServicer(client_manager)
    add_servicer_to_server_fn = add_FlowerServiceServicer_to_server

    server = generic_create_grpc_server(
        servicer_and_add_fn=(servicer, add_servicer_to_server_fn),
        server_address=server_address,
        max_concurrent_workers=max_concurrent_workers,
        max_message_length=max_message_length,
        keepalive_time_ms=keepalive_time_ms,
        certificates=certificates,
    )

    # add a servicer that communicates with the clients for SL
    servicer = ServerSegmentServicer(server_model_manager)
    server_segment_pb2_grpc.add_ServerSegmentServicer_to_server(servicer, server)
    server.start()

    return server


def start_server(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    server_address: str = ADDRESS_FLEET_API_GRPC_BIDI,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[SlStrategy] = None,
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
    grpc_server = start_grpc_server(
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
    hist = run_fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Stop the gRPC server
    grpc_server.stop(grace=1)

    event(EventType.START_SERVER_LEAVE)

    return hist
