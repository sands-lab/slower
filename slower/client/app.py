import time
from logging import INFO
from typing import Optional, Union, Callable
from slower.client.client import Client
from slower.client.typing import ClientFn

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.flower import Fwd, Bwd
from flwr.client.workload_state import WorkloadState
from flwr.common.logger import log, warn_experimental_feature

from slower.client.grpc.grpc_client import GrpcClient
from slower.client.grpc.connection import init_connection


def _check_actionable_client(
    client: Optional[Client], client_fn: Optional[ClientFn]
) -> None:
    if client_fn is None and client is None:
        raise Exception("Both `client_fn` and `client` are `None`, but one is required")

    if client_fn is not None and client is not None:
        raise Exception(
            "Both `client_fn` and `client` are provided, but only one is allowed"
        )


def start_client(
    *,
    server_address: str,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
) -> None:

    event(EventType.START_CLIENT_ENTER)
    _start_client_internal(
        server_address=server_address,
        load_flower_callable_fn=None,
        client_fn=client_fn,
        client=client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        insecure=insecure,
        transport=transport,
    )
    event(EventType.START_CLIENT_LEAVE)


def _start_client_internal(
    *,
    server_address: str,
    load_flower_callable_fn: Optional[Callable[[], GrpcClient]] = None,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
) -> None:
    """Start a Flower client node which connects to a Flower server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    load_flower_callable_fn : Optional[Callable[[], Flower]] (default: None)
        A function that can be used to load a `Flower` callable instance.
    client_fn : Optional[ClientFn]
        A callable that instantiates a Client. (default: None)
    client : Optional[flwr.client.Client]
        An implementation of the abstract base
        class `flwr.client.Client` (default: None)
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : bool (default: True)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)
    """
    if insecure is None:
        insecure = root_certificates is None

    # Initialize connection context manager
    connection, address = init_connection(server_address)

    if load_flower_callable_fn is None:
        _check_actionable_client(client, client_fn)

        if client_fn is None:
            # Wrap `Client` instance in `client_fn`
            def single_client_factory(
                cid: str,  # pylint: disable=unused-argument
            ) -> Client:
                if client is None:  # Added this to keep mypy happy
                    raise Exception(
                        "Both `client_fn` and `client` are `None`, but one is required"
                    )
                return client  # Always return the same instance

            client_fn = single_client_factory

        def _load_app() -> GrpcClient:
            return GrpcClient(client_fn=client_fn)

        load_flower_callable_fn = _load_app
    else:
        warn_experimental_feature("`load_flower_callable_fn`")

    # At this point, only `load_flower_callable_fn` should be used
    # Both `client` and `client_fn` must not be used directly

    while True:
        sleep_duration: int = 0
        with connection(
            address,
            insecure,
            grpc_max_message_length,
            root_certificates,
        ) as conn:
            receive, send, stub = conn

            while True:
                # Receive
                task_ins = receive()
                if task_ins is None:
                    time.sleep(3)  # Wait for 3s before asking again
                    continue

                # Handle control message
                task_res, sleep_duration = handle_control_message(task_ins=task_ins)
                if task_res:
                    send(task_res)
                    break

                # Load app
                app: GrpcClient = load_flower_callable_fn()

                # Handle task message
                fwd_msg: Fwd = Fwd(
                    task_ins=task_ins,
                    state=WorkloadState(state={}),
                )
                bwd_msg: Bwd = app(fwd=fwd_msg, stub=stub)

                # Send
                send(bwd_msg.task_res)

        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)
