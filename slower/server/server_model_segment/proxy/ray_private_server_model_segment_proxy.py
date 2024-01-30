from typing import Optional

from flwr.common import GetParametersRes

from slower.server.server_model_segment.server_model_segment import ServerModelSegment
from slower.common import (
    ServerModelSegmentEvaluateIns,
    ServerModelSegmentFitIns,
    BatchPredictionIns,
    BatchPredictionRes,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes
)

from slower.server.server_model_segment.proxy.server_model_segment_proxy import (
    ServerModelSegmentProxy
)



class RayPrivateServerModelSegmentProxy(ServerModelSegmentProxy):
    """Slower server model proxy which which acts as a POPO inside the ray actor (the client)."""

    def __init__(
        self,
        server_model_segment: ServerModelSegment
    ):
        super().__init__()
        self.server_model_segment = server_model_segment

    def get_parameters(
        self,
        timeout: Optional[float]
    ) -> GetParametersRes:
        return self.server_model_segment.get_parameters()

    def configure_fit(
        self, ins: ServerModelSegmentFitIns, timeout: Optional[float]
    ):
        self.server_model_segment.configure_fit(ins)

    def configure_evaluate(
        self, ins: ServerModelSegmentEvaluateIns, timeout: Optional[float]
    ):
        self.server_model_segment.configure_evaluate(ins)

    def serve_prediction_request(
        self,
        batch_data: BatchPredictionIns,
        timeout: Optional[float]
    ) -> BatchPredictionRes:
        return self.server_model_segment.serve_prediction_request(batch_data)

    def serve_gradient_update_request(
        self,
        batch_data: GradientDescentDataBatchIns,
        timeout: Optional[float]
    ) -> GradientDescentDataBatchRes:
        return self.server_model_segment.serve_gradient_update_request(batch_data)
