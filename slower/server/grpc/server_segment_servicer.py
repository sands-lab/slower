import grpc

import slower.proto.server_segment_pb2_grpc as server_segment_pb2_grpc
import slower.proto.server_segment_pb2 as server_segment_pb2
from slower.server.server_model_segment.manager.server_model_segment_manager import ServerModelSegmentManager
from slower.common import (
    GradientDescentDataBatchIns,
    BatchPredictionIns,
)


class ServerSegmentServicer(server_segment_pb2_grpc.ServerSegmentServicer):
    def __init__(self, server_model_segment_manager: ServerModelSegmentManager) -> None:
        super().__init__()
        self.server_model_segment_manager = server_model_segment_manager

    def ServePredictionRequest(
        self,
        request: server_segment_pb2.BatchPredictionIns,
        context
    ) -> server_segment_pb2.BatchPredictionRes:
        proxy = self.server_model_segment_manager.get_server_model_segment_proxy(context.peer())
        data = BatchPredictionIns(embeddings=request.embeddings)
        res = proxy.serve_prediction_request(data, None)
        return server_segment_pb2.BatchPredictionRes(predictions=res.predictions)

    def ServeGradientUpdateRequest(
        self,
        request: server_segment_pb2.GradientDescentDataBatchIns,
        context
    ) -> server_segment_pb2.GradientDescentDataBatchRes:
        # routing based on IP address
        proxy = self.server_model_segment_manager.get_server_model_segment_proxy(context.peer())
        data = GradientDescentDataBatchIns(embeddings=request.embeddings, labels=request.labels)
        res = proxy.serve_gradient_update_request(data, None)
        return server_segment_pb2.GradientDescentDataBatchRes(gradient=res.gradient)
