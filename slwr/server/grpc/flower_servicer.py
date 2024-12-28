import uuid
from typing import Callable, Iterator

import grpc
from iterators import TimeoutIterator

from flwr.proto import transport_pb2_grpc  # pylint: disable=E0611
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)
from flwr.server.client_manager import ClientManager
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import (
    GrpcBridge,
    InsWrapper,
    ResWrapper,
)
from flwr.server.superlink.fleet.grpc_bidi.grpc_client_proxy import GrpcClientProxy


def default_bridge_factory() -> GrpcBridge:
    """Return GrpcBridge instance."""
    return GrpcBridge()


def default_grpc_client_proxy_factory(cid: str, bridge: GrpcBridge) -> GrpcClientProxy:
    """Return GrpcClientProxy instance."""
    return GrpcClientProxy(cid=cid, bridge=bridge)


def register_client_proxy(
    client_manager: ClientManager,
    client_proxy: GrpcClientProxy,
    context: grpc.ServicerContext,
) -> bool:
    """Try registering GrpcClientProxy with ClientManager."""
    return client_manager.register(client_proxy)


class FlowerServiceServicer(transport_pb2_grpc.FlowerServiceServicer):
    """Had to copy-paste the whole class because of a bug in the `add_callback` function
    when using asyncio
    There's an open github issue for this bug:
    https://github.com/grpc/grpc/issues/38346
    """

    def __init__(
        self,
        client_manager: ClientManager,
        grpc_bridge_factory: Callable[[], GrpcBridge] = default_bridge_factory,
        grpc_client_proxy_factory: Callable[
            [str, GrpcBridge], GrpcClientProxy
        ] = default_grpc_client_proxy_factory,
    ) -> None:
        self.client_manager: ClientManager = client_manager
        self.grpc_bridge_factory = grpc_bridge_factory
        self.client_proxy_factory = grpc_client_proxy_factory

    def Join(  # pylint: disable=invalid-name
        self,
        request_iterator: Iterator[ClientMessage],
        context: grpc.ServicerContext,
    ) -> Iterator[ServerMessage]:
        cid: str = uuid.uuid4().hex
        bridge = self.grpc_bridge_factory()
        client_proxy = self.client_proxy_factory(cid, bridge)
        is_success = register_client_proxy(self.client_manager, client_proxy, context)

        if is_success:
            # Get iterators
            client_message_iterator = TimeoutIterator(
                iterator=request_iterator, reset_on_next=True
            )
            ins_wrapper_iterator = bridge.ins_wrapper_iterator()

            # All messages will be pushed to client bridge directly
            while True:
                try:
                    # Get ins_wrapper from bridge and yield server_message
                    ins_wrapper: InsWrapper = next(ins_wrapper_iterator)
                    yield ins_wrapper.server_message

                    # Set current timeout, might be None
                    if ins_wrapper.timeout is not None:
                        client_message_iterator.set_timeout(ins_wrapper.timeout)

                    # Wait for client message
                    client_message = next(client_message_iterator)

                    if client_message is client_message_iterator.get_sentinel():
                        # Important: calling `context.abort` in gRPC always
                        # raises an exception so that all code after the call to
                        # `context.abort` will not run. If subsequent code should
                        # be executed, the `rpc_termination_callback` can be used
                        # (as shown in the `register_client` function).
                        details = f"Timeout of {ins_wrapper.timeout}sec was exceeded."
                        context.abort(
                            code=grpc.StatusCode.DEADLINE_EXCEEDED,
                            details=details,
                        )
                        return

                    bridge.set_res_wrapper(
                        res_wrapper=ResWrapper(client_message=client_message)
                    )
                except Exception:
                    break

        client_proxy.bridge.close()
        self.client_manager.unregister(client_proxy)
