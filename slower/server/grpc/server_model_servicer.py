from slower.proto import server_model_pb2_grpc
from slower.proto import server_model_pb2
from slower.server.request_handler.request_handler import RequestHandler
from slower.common import BatchData, ControlCode
from slower.common.serde import (
    control_code_from_proto,
    control_code_to_proto,
    from_grpc_format,
    to_grpc_format
)


# pylint: disable=no-member
class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):
    def __init__(self, request_handler: RequestHandler) -> None:
        super().__init__()
        self.request_handler = request_handler

    def _from_grpc(self, server_model, request: server_model_pb2.BatchData):
        method_name = request.method
        data = BatchData(
            data = from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code)
        )
        method = getattr(server_model, method_name)
        return method, data

    def _to_grpc(self, data: BatchData):
        return server_model_pb2.BatchData(
            method="",
            data=to_grpc_format(data.data),
            control_code=control_code_to_proto(data.control_code)
        )

    def BlockingRequest(
        self,
        request: server_model_pb2.BatchData,
        context
    ):
        with self.request_handler.get_server_model(context.peer()) as server_model:
            method, data = self._from_grpc(server_model, request)
            res = method(data)
        return self._to_grpc(res)

    def StreamingRequests(self, request_iterator, context):
        for request in request_iterator:
            if control_code_from_proto(request.control_code) == ControlCode.DO_CLOSE_STREAM:
                break
            with self.request_handler.get_server_model(context.peer()) as server_model:
                method, data = self._from_grpc(server_model, request)
                method(data)

        res = server_model.get_synchronization_result()
        return server_model_pb2.BatchData(
            data=to_grpc_format(res.data),
            control_code=control_code_to_proto(res.control_code),
        )
