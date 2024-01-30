from typing import Optional

from flwr.common import GetParametersRes

from slower.common import (
    BatchPredictionIns,
    BatchPredictionRes,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    ServerModelSegmentEvaluateIns,
    ServerModelSegmentFitIns,
)
from slower.server.server_model_segment.proxy.server_model_segment_proxy import ServerModelSegmentProxy
import slower.proto.server_segment_pb2_grpc as server_segment_pb2_grpc
import slower.proto.server_segment_pb2 as server_segment_pb2


class ClientException(Exception):
    """Exception indicating that the client invoked a method it should not have. For instance,
    server model segment configuration should be taken care of by the server, and any attempt
    at doing so by the client will result in an exception. The client may only use this proxy
    for computing predictions and for requesting gradient updates."""

class GrpcServerModelSegmentProxy(ServerModelSegmentProxy):

    def __init__(self, stub: server_segment_pb2_grpc.ServerSegmentStub, cid: str) -> None:
        super().__init__()
        self.stub = stub
        self.cid = cid

    def serve_prediction_request(
        self,
        batch_data: BatchPredictionIns,
        timeout: Optional[float]
    ) -> BatchPredictionRes:
        ins = server_segment_pb2.BatchPredictionIns(embeddings=batch_data.embeddings)
        res: server_segment_pb2.BatchPredictionRes = self.stub.ServePredictionRequest(ins)
        return BatchPredictionRes(predictions=res.predictions)

    def serve_gradient_update_request(
        self,
        batch_data: GradientDescentDataBatchIns,
        timeout: Optional[float]
    ) -> GradientDescentDataBatchRes:
        ins = server_segment_pb2.GradientDescentDataBatchIns(
            embeddings=batch_data.embeddings,
            labels=batch_data.labels
        )
        res: server_segment_pb2.GradientDescentDataBatchRes = self.stub.ServeGradientUpdateRequest(ins)
        return GradientDescentDataBatchRes(gradient=res.gradient)

    def get_parameters(
        self,
        timeout: Optional[float]
    ) -> GetParametersRes:
        raise ClientException("Client tried to get parameters from the server")

    def configure_fit(
        self,
        ins: ServerModelSegmentFitIns,
        timeout: Optional[float]
    ) -> None:
        raise ClientException("Client tried to configure server fit")


    def configure_evaluate(
        self,
        ins: ServerModelSegmentEvaluateIns,
        timeout: Optional[float]
    ) -> None:
        raise ClientException("Client tried to configure server evaluate")
