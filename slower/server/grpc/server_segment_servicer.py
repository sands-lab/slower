from typing import Iterable

from slower.proto import server_segment_pb2_grpc
from slower.proto import server_segment_pb2
from slower.server.server_model.manager.server_model_manager import (
    ServerModelManager
)
from slower.common import GradientDescentDataBatchIns, BatchPredictionIns, DataBatchForward, DataBatchBackward
from slower.common.serde import control_code_from_proto, control_code_to_proto


# pylint: disable=no-member
class ServerSegmentServicer(server_segment_pb2_grpc.ServerSegmentServicer):
    def __init__(self, server_model_manager: ServerModelManager) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager

    def ServePredictionRequest(
        self,
        request: server_segment_pb2.BatchPredictionIns,
        context
    ) -> server_segment_pb2.BatchPredictionRes:
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        data = BatchPredictionIns(
            embeddings=request.embeddings,
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.serve_prediction_request(data, None)

        return server_segment_pb2.BatchPredictionRes(
            predictions=res.predictions,
            control_code=control_code_to_proto(res.control_code)
        )

    def ServeGradientUpdateRequest(
        self,
        request: server_segment_pb2.GradientDescentDataBatchIns,
        context
    ) -> server_segment_pb2.GradientDescentDataBatchRes:

        cid = context.peer()
        proxy = self.server_model_manager.get_server_model_proxy(cid)

        data = GradientDescentDataBatchIns(
            embeddings=request.embeddings,
            labels=request.labels,
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.serve_gradient_update_request(data, None)

        return server_segment_pb2.GradientDescentDataBatchRes(
            gradient=res.gradient,
            control_code=control_code_to_proto(res.control_code)
        )

    def UpdateServerSideModelRequests(
        self,
        requests_iterator: Iterable[server_segment_pb2.GradientDescentDataBatchIns],
        context
    ):
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        _iter = (GradientDescentDataBatchIns(
            embeddings=req.embeddings,
            labels=req.labels,
            control_code=control_code_from_proto(req.control_code)
        ) for req in requests_iterator)
        res = proxy.update_server_side_model_requests(_iter)
        return server_segment_pb2.UpdateServerSideModelRes(
            control_code=control_code_to_proto(res.control_code)
        )

    def UForward(self, request: server_segment_pb2.DataBatchForward, context):
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        batch = DataBatchForward(
            embeddings=request.embeddings,
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.u_forward(batch=batch)
        return server_segment_pb2.DataBatchForward(
            embeddings=res.embeddings,
            control_code=control_code_to_proto(res.control_code)
        )

    def UBackward(self, request: server_segment_pb2.DataBatchBackward, context):
        proxy = self.server_model_manager.get_server_model_proxy(context.peer())
        batch_gradient = DataBatchBackward(
            gradient=request.gradient,
            control_code=control_code_from_proto(request.control_code)
        )
        res = proxy.u_backward(batch_gradient=batch_gradient)
        return server_segment_pb2.DataBatchBackward(
            gradient=res.gradient,
            control_code=control_code_to_proto(res.control_code)
        )
