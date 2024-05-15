from typing import Iterator

from slower.proto import server_model_pb2_grpc
from slower.proto import server_model_pb2
from slower.server.server_model.manager.server_model_manager import ServerModelManager
from slower.common import (
    GradientDescentDataBatchIns,
    BatchPredictionIns,
    DataBatchForward,
    DataBatchBackward,
)
from slower.common.serde import control_code_from_proto, control_code_to_proto


# pylint: disable=no-member
class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):
    def __init__(self, server_model_manager: ServerModelManager) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager

    def ServePredictionRequest(
        self,
        request: server_model_pb2.BatchPredictionIns,
        context
    ) -> server_model_pb2.BatchPredictionRes:
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        data = BatchPredictionIns(
            embeddings=request.embeddings,
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.serve_prediction_request(data, None)

        return server_model_pb2.BatchPredictionRes(
            predictions=res.predictions,
            control_code=control_code_to_proto(res.control_code)
        )

    def ServeGradientUpdateRequest(
        self,
        request: server_model_pb2.GradientDescentDataBatchIns,
        context
    ) -> server_model_pb2.GradientDescentDataBatchRes:

        cid = context.peer()
        proxy = self.server_model_manager.get_server_model_proxy(cid)

        data = GradientDescentDataBatchIns(
            embeddings=request.embeddings,
            labels=request.labels,
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.serve_gradient_update_request(data, None)

        return server_model_pb2.GradientDescentDataBatchRes(
            gradient=res.gradient,
            control_code=control_code_to_proto(res.control_code)
        )

    def UpdateServerModelRequests(
        self,
        requests_iterator: Iterator[server_model_pb2.GradientDescentDataBatchIns],
        context
    ) -> server_model_pb2.UpdateServerModelRes:
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        _iter = (GradientDescentDataBatchIns(
            embeddings=req.embeddings,
            labels=req.labels,
            control_code=control_code_from_proto(req.control_code)
        ) for req in requests_iterator)
        proxy._update_server_model_requests(_iter)
        res = proxy._get_synchronization_result()
        return server_model_pb2.UpdateServerModelRes(
            control_code=control_code_to_proto(res.control_code),
            result=res.result
        )

    def UForward(self, request: server_model_pb2.DataBatchForward, context):
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        batch = DataBatchForward(
            embeddings=request.embeddings,
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.u_forward(batch=batch)
        return server_model_pb2.DataBatchForward(
            embeddings=res.embeddings,
            control_code=control_code_to_proto(res.control_code)
        )

    def UBackward(self, request: server_model_pb2.DataBatchBackward, context):
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        batch_gradient = DataBatchBackward(
            gradient=request.gradient,
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.u_backward(batch_gradient=batch_gradient)
        return server_model_pb2.DataBatchBackward(
            gradient=res.gradient,
            control_code=control_code_to_proto(res.control_code)
        )
