from typing import Optional
from abc import ABC, abstractmethod

from flwr.common import GetParametersRes

from slower.common import (
    BatchPredictionIns,
    BatchPredictionRes,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    ServerModelSegmentEvaluateIns,
    ServerModelSegmentFitIns,
)


class ServerModelSegmentProxy(ABC):

    @abstractmethod
    def serve_prediction_request(
        self,
        batch_data: BatchPredictionIns,
        timeout: Optional[float]
    ) -> BatchPredictionRes:
        """Compute the final predictions"""

    @abstractmethod
    def serve_gradient_update_request(
        self,
        batch_data: GradientDescentDataBatchIns,
        timeout: Optional[float]
    ) -> GradientDescentDataBatchRes:
        """Compute the loss, and backpropagate it and send to the client the gradient info"""

    @abstractmethod
    def get_parameters(
        self,
        timeout: Optional[float]
    ) -> GetParametersRes:
        """Return the weights of the server model"""

    @abstractmethod
    def configure_fit(
        self,
        ins: ServerModelSegmentFitIns,
        timeout: Optional[float]
    ) -> None:
        """Configure the server-side model prior to training"""

    @abstractmethod
    def configure_evaluate(
        self,
        ins: ServerModelSegmentEvaluateIns,
        timeout: Optional[float]
    ) -> None:
        """Configure the server-side model prior to evaluation"""

    def serve_prediction_request_wrapper(self, embeddings: bytes, timeout: float) -> bytes:
        ins = BatchPredictionIns(embeddings=embeddings)
        res = self.serve_prediction_request(ins, timeout)
        return res.predictions

    def serve_gradient_update_request_wrapper(
        self,
        embeddings: bytes,
        labels: bytes,
        timeout: float
    ) -> bytes:
        ins = GradientDescentDataBatchIns(embeddings=embeddings, labels=labels)
        res = self.serve_gradient_update_request(ins, timeout)
        return res.gradient
