import time
from logging import INFO, WARN, DEBUG, ERROR
from typing import Optional, Union, Callable, Tuple

from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.constant import TRANSPORT_TYPE_GRPC_BIDI
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.app import _AppStateTracker, _check_actionable_client
from flwr.client.node_state import NodeState
from flwr.client.client_app import LoadClientAppError
from flwr.common.retry_invoker import RetryInvoker, RetryState, exponential
from flwr.common.message import Error
from flwr.common.constant import ErrorCode
from flwr.common.logger import log

from slwr.client.client import Client
from slwr.client.grpc.connection import init_connection
from slwr.client.client_app import ClientApp


ClientFn = Callable[[str], Client]



# pylint: disable=too-many-arguments
def start_client(
    *,
    server_address: str,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = TRANSPORT_TYPE_GRPC_BIDI,
    authentication_keys: Optional[
        Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
) -> None:

    event(EventType.START_CLIENT_ENTER)
    _start_client_internal(
        server_address=server_address,
        load_client_app_fn=None,
        client_fn=client_fn,
        client=client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        insecure=insecure,
        transport=transport,
        authentication_keys=authentication_keys,
        max_retries=max_retries,
        max_wait_time=max_wait_time,
    )
    event(EventType.START_CLIENT_LEAVE)


# pylint: disable=too-many-arguments,too-many-arguments,too-many-locals
def _start_client_internal(
    *,
    server_address: str,
    load_client_app_fn: Optional[Callable[[], ClientApp]] = None,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
    authentication_keys: Optional[
        Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
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
    """
    if insecure is None:
        insecure = root_certificates is None

    if load_client_app_fn is None:
        _check_actionable_client(client, client_fn)

        if client_fn is None:
            # Wrap `Client` instance in `client_fn`
            def single_client_factory(
                cid: str,  # pylint: disable=unused-argument
            ) -> Client:
                if client is None:  # Added this to keep mypy happy
                    raise ValueError(
                        "Both `client_fn` and `client` are `None`, but one is required"
                    )
                return client  # Always return the same instance

            client_fn = single_client_factory

        def _load_client_app() -> ClientApp:
            return ClientApp(client_fn=client_fn)

        load_client_app_fn = _load_client_app

    # At this point, only `load_client_app_fn` should be used
    # Both `client` and `client_fn` must not be used directly

    # Initialize connection context manager
    connection, address, connection_error_type = init_connection(
        transport, server_address
    )

    app_state_tracker = _AppStateTracker()

    def _on_sucess(retry_state: RetryState) -> None:
        app_state_tracker.is_connected = True
        if retry_state.tries > 1:
            log(
                INFO,
                "Connection successful after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    def _on_backoff(retry_state: RetryState) -> None:
        app_state_tracker.is_connected = False
        if retry_state.tries == 1:
            log(WARN, "Connection attempt failed, retrying...")
        else:
            log(
                DEBUG,
                "Connection attempt failed, retrying in %.2f seconds",
                retry_state.actual_wait,
            )

    retry_invoker = RetryInvoker(
        wait_gen_factory=exponential,
        recoverable_exceptions=connection_error_type,
        max_tries=max_retries,
        max_time=max_wait_time,
        on_giveup=lambda retry_state: (
            log(
                WARN,
                "Giving up reconnection after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )
            if retry_state.tries > 1
            else None
        ),
        on_success=_on_sucess,
        on_backoff=_on_backoff,
    )

    node_state = NodeState()
    # At this point, only `load_flower_callable_fn` should be used
    # Both `client` and `client_fn` must not be used directly

    while not app_state_tracker.interrupt:
        sleep_duration: int = 0
        with connection(
            address,
            insecure,
            retry_invoker,
            grpc_max_message_length,
            root_certificates,
            authentication_keys,
        ) as conn:
            receive, send, server_model_stub, create_node, delete_node, _ = conn

            if create_node is not None:
                create_node()  # pylint: disable=not-callable

            app_state_tracker.register_signal_handler()
            while not app_state_tracker.interrupt:
                try:
                    # Receive
                    message = receive()
                    if message is None:
                        time.sleep(3)  # Wait for 3s before asking again
                        continue

                    log(INFO, "")
                    log(
                        INFO,
                        "Received: %s message %s",
                        message.metadata.message_type,
                        message.metadata.message_id,
                    )

                    # Handle control message
                    out_message, sleep_duration = handle_control_message(message)
                    if out_message:
                        send(out_message)
                        break

                    # Register context for this run
                    node_state.register_context(run_id=message.metadata.run_id)

                    # Retrieve context for this run
                    context = node_state.retrieve_context(
                        run_id=message.metadata.run_id
                    )

                    # Create an error reply message that will never be used to prevent
                    # the used-before-assignment linting error
                    reply_message = message.create_error_reply(
                        error=Error(code=ErrorCode.UNKNOWN, reason="Unknown")
                    )

                    # Handle app loading and task message
                    try:
                        # Load ClientApp instance
                        client_app: ClientApp = load_client_app_fn()

                        # Execute ClientApp
                        reply_message = client_app(
                            message=message,
                            context=context,
                            server_model_stub=server_model_stub
                        )
                    except Exception as ex:  # pylint: disable=broad-exception-caught

                        # Legacy grpc-bidi
                        if transport in ["grpc-bidi", None]:
                            log(ERROR, "Client raised an exception.", exc_info=ex)
                            # Raise exception, crash process
                            raise ex

                        # Don't update/change NodeState

                        e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
                        # Ex fmt: "<class 'ZeroDivisionError'>:<'division by zero'>"
                        reason = str(type(ex)) + ":<'" + str(ex) + "'>"
                        exc_entity = "ClientApp"
                        if isinstance(ex, LoadClientAppError):
                            reason = (
                                "An exception was raised when attempting to load "
                                "`ClientApp`"
                            )
                            e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION
                            exc_entity = "SuperNode"

                        if not app_state_tracker.interrupt:
                            log(
                                ERROR, "%s raised an exception", exc_entity, exc_info=ex
                            )

                        # Create error message
                        reply_message = message.create_error_reply(
                            error=Error(code=e_code, reason=reason)
                        )
                    else:
                        # No exception, update node state
                        node_state.update_context(
                            run_id=message.metadata.run_id,
                            context=context,
                        )

                    # Send
                    send(reply_message)
                    log(INFO, "Sent reply")

                except StopIteration:
                    sleep_duration = 0
                    break

            # Unregister node
            if delete_node is not None and app_state_tracker.is_connected:
                delete_node()  # pylint: disable=not-callable

        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            del app_state_tracker
            break

        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)
