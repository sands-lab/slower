from flwr.client.flower import Fwd, Bwd
from flwr.client.workload_state import WorkloadState
from flwr.proto.task_pb2 import SecureAggregation, Task, TaskIns, TaskRes
from flwr.client.message_handler.task_handler import (
    get_server_message_from_task_ins,
    wrap_client_message_in_task_res,
)
from flwr.client.message_handler.message_handler import (
    UnknownServerMessage,
    UnexpectedServerMessage,
    _fit,
    _evaluate,
    _get_parameters,
    _get_properties
)
from flwr.client.secure_aggregation import SecureAggregationHandler
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.common import serde

from slower.proto import server_segment_pb2_grpc
from slower.server.server_model.proxy.grpc_server_model_proxy import (
    GrpcServerModelProxy
)
from slower.client.typing import ClientFn


# pylint: disable=too-few-public-methods
class GrpcClient:

    def __init__(
        self,
        client_fn: ClientFn,  # Only for backward compatibility
    ) -> None:
        self.client_fn = client_fn

    def __call__(self, fwd: Fwd, stub: server_segment_pb2_grpc.ServerSegmentStub) -> Bwd:
        """."""
        server_model_proxy = GrpcServerModelProxy(stub, "-1")
        # Execute the task
        task_res = handle(
            client_fn=self.client_fn,
            task_ins=fwd.task_ins,
            server_model_proxy=server_model_proxy,
        )
        return Bwd(
            task_res=task_res,
            state=WorkloadState(state={}),
        )


def handle(
    client_fn: ClientFn,
    task_ins: TaskIns,
    server_model_proxy: GrpcServerModelProxy
) -> TaskRes:
    """Handle incoming TaskIns from the server.

    Parameters
    ----------
    client_fn : ClientFn
        A callable that instantiates a Client.
    task_ins: TaskIns
        The task instruction coming from the server, to be processed by the client.

    Returns
    -------
    task_res : TaskRes
        The task response that should be returned to the server.
    """
    server_msg = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=False)
    if server_msg is None:
        # Instantiate the client
        client = client_fn("-1")
        # Secure Aggregation
        if task_ins.task.HasField("sa") and isinstance(
            client, SecureAggregationHandler
        ):
            # pylint: disable-next=invalid-name
            named_values = serde.named_values_from_proto(task_ins.task.sa.named_values)
            res = client.handle_secure_aggregation(named_values)
            task_res = TaskRes(
                task_id="",
                group_id="",
                workload_id=0,
                task=Task(
                    ancestry=[],
                    sa=SecureAggregation(named_values=serde.named_values_to_proto(res)),
                ),
            )
            return task_res
        raise NotImplementedError()
    client_msg = handle_legacy_message(client_fn, server_msg, server_model_proxy)
    task_res = wrap_client_message_in_task_res(client_msg)
    return task_res


def handle_legacy_message(
    client_fn: ClientFn,
    server_msg: ServerMessage,
    server_model_proxy: GrpcServerModelProxy
) -> ClientMessage:
    """Handle incoming messages from the server.

    Parameters
    ----------
    client_fn : ClientFn
        A callable that instantiates a Client.
    server_msg: ServerMessage
        The message coming from the server, to be processed by the client.

    Returns
    -------
    client_msg : ClientMessage
        The result message that should be returned to the server.
    """
    field = server_msg.WhichOneof("msg")

    # Must be handled elsewhere
    if field == "reconnect_ins":
        raise UnexpectedServerMessage()

    # Instantiate the client
    client = client_fn("-1").to_client()
    client.set_server_model_proxy(server_model_proxy)
    # Execute task
    if field == "get_properties_ins":
        return _get_properties(client, server_msg.get_properties_ins)
    if field == "get_parameters_ins":
        return _get_parameters(client, server_msg.get_parameters_ins)
    if field == "fit_ins":
        return _fit(client, server_msg.fit_ins)
    if field == "evaluate_ins":
        return _evaluate(client, server_msg.evaluate_ins)
    raise UnknownServerMessage()
