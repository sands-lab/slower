from typing import Iterator

from slower.proto import server_model_pb2_grpc
from slower.proto import server_model_pb2
from slower.server.server_model.manager.server_model_manager import ServerModelManager
from slower.common import BatchData
from slower.common.serde import (
    control_code_from_proto,
    control_code_to_proto,
    from_grpc_format,
    to_grpc_format
)


# pylint: disable=no-member
class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):
    def __init__(self, server_model_manager: ServerModelManager) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager

    def ServePredictionRequest(
        self,
        request: server_model_pb2.BatchData,
        context
    ) -> server_model_pb2.BatchData:
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        data = BatchData(
            data=from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.serve_prediction_request(data, None)

        return server_model_pb2.BatchData(
            data=to_grpc_format(res.data),
            control_code=control_code_to_proto(res.control_code)
        )

    def ServeGradientUpdateRequest(
        self,
        request: server_model_pb2.BatchData,
        context
    ) -> server_model_pb2.BatchData:

        cid = context.peer()
        proxy = self.server_model_manager.get_server_model_proxy(cid)

        data = BatchData(
            data=from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code),
        )
        res = proxy.serve_gradient_update_request(data, None)

        return server_model_pb2.BatchData(
            data=to_grpc_format(res.data),
            control_code=control_code_to_proto(res.control_code)
        )

    def UpdateServerModelRequests(
        self,
        request_iterator: Iterator[server_model_pb2.BatchData],
        context
    ) -> server_model_pb2.BatchData:
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        _iter = (BatchData(
            data=from_grpc_format(req.data),
            control_code=control_code_from_proto(req.control_code)
        ) for req in request_iterator)
        proxy._update_server_model_requests(_iter)
        res = proxy._get_synchronization_result()
        return server_model_pb2.BatchData(
            data=to_grpc_format(res.data),
            control_code=control_code_to_proto(res.control_code),
        )

    def UForward(
        self,
        request: server_model_pb2.BatchData,
        context
    ) -> server_model_pb2.BatchData:
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        batch_data = BatchData(
            data=from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.u_forward(batch_data=batch_data)
        return server_model_pb2.BatchData(
            data=to_grpc_format(res.data),
            control_code=control_code_to_proto(res.control_code)
        )

    def UBackward(
        self,
        request: server_model_pb2.BatchData,
        context
    ) -> server_model_pb2.BatchData:
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        batch_data = BatchData(
            data=from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.u_backward(batch_data=batch_data)
        return server_model_pb2.BatchData(
            data=to_grpc_format(res.data),
            control_code=control_code_to_proto(res.control_code)
        )
