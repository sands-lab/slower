from typing import Optional, Iterator
from abc import ABC, abstractmethod

import numpy as np
from flwr.common import GetParametersRes, ndarray_to_bytes, bytes_to_ndarray

from slower.common import (
    BatchPredictionIns,
    BatchPredictionRes,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ControlCode,
    UpdateServerModelRes,
    DataBatchForward,
    DataBatchBackward
)


class ServerModelProxy(ABC):

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
        ins: ServerModelFitIns,
        timeout: Optional[float]
    ) -> None:
        """Configure the server-side model prior to training"""

    @abstractmethod
    def configure_evaluate(
        self,
        ins: ServerModelEvaluateIns,
        timeout: Optional[float]
    ) -> None:
        """Configure the server-side model prior to evaluation"""

    def numpy_serve_prediction_request(
        self,
        embeddings: np.ndarray,
        timeout: Optional[float] = None
    ) -> np.ndarray:
        res = self.bytes_serve_prediction_request(
            embeddings=ndarray_to_bytes(embeddings),
            timeout=timeout
        )
        return bytes_to_ndarray(res)

    def bytes_serve_prediction_request(
        self,
        embeddings: bytes,
        timeout: Optional[float] = None
    ):
        ins = BatchPredictionIns(
            embeddings=embeddings,
            control_code=ControlCode.OK
        )
        res = self.serve_prediction_request(ins, timeout)
        return res.predictions

    def numpy_serve_gradient_update_request(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        timeout: Optional[float] = None
    ) -> np.ndarray:
        res = self.bytes_serve_gradient_update_request(
            embeddings=ndarray_to_bytes(embeddings),
            labels=ndarray_to_bytes(labels),
            timeout=timeout
        )
        return bytes_to_ndarray(res)

    def bytes_serve_gradient_update_request(
        self,
        embeddings: bytes,
        labels: bytes,
        timeout: Optional[float] = None
    ):
        ins = GradientDescentDataBatchIns(
            embeddings=embeddings,
            labels=labels,
            control_code=ControlCode.OK
        )
        res = self.serve_gradient_update_request(ins, timeout)
        return res.gradient

    @abstractmethod
    def update_server_model(self, batch: GradientDescentDataBatchIns):
        """Receives a single batch and sends it to the server for being processed"""

    def numpy_update_server_model(self, embeddings, labels):
        ins = GradientDescentDataBatchIns(
            embeddings=ndarray_to_bytes(embeddings),
            labels=ndarray_to_bytes(labels),
            control_code=ControlCode.OK
        )
        self.update_server_model(ins)

    def _update_server_model_requests(
        self, batches_iterator: Iterator[GradientDescentDataBatchIns]
    ) -> None:

        for batch in batches_iterator:
            if batch.control_code == ControlCode.DO_CLOSE_STREAM:
                break
            self.serve_gradient_update_request(batch, None)

    def bytes_update_server_model(
        self,
        embeddings: bytes,
        labels: bytes
    ):
        ins = GradientDescentDataBatchIns(
            embeddings=embeddings,
            labels=labels,
            control_code=ControlCode.OK
        )
        self.update_server_model(ins)

    @abstractmethod
    def u_forward(self, batch: DataBatchForward) -> DataBatchForward:
        """Invoke the forward pass on the server-side model"""

    def numpy_u_forward(self, embeddings: np.ndarray):
        res = self.bytes_u_forward(ndarray_to_bytes(embeddings))
        return bytes_to_ndarray(res)

    def numpy_u_backward(self, gradient: np.ndarray, blocking=True):
        res = self.bytes_u_backward(ndarray_to_bytes(gradient), blocking=blocking)
        return bytes_to_ndarray(res)

    def bytes_u_forward(self, embeddings):
        batch = DataBatchForward(embeddings=embeddings, control_code=ControlCode.OK)
        res = self.u_forward(batch=batch)
        return res.embeddings

    def bytes_u_backward(self, gradient, blocking=True):
        batch_gradient = DataBatchBackward(
            gradient=gradient,
            control_code=ControlCode.OK
        )
        res = self.u_backward(batch_gradient=batch_gradient, blocking=blocking)
        if blocking:
            return res.gradient
        else:
            assert res is None

    @abstractmethod
    def u_backward(self, batch_gradient: DataBatchBackward, blocking=True) -> DataBatchBackward:
        """Invoke the backward pass on the server. If blocking is set to true, the client will wait
        until it gets a response from the server, otherwise the client will just send the data"""

    @abstractmethod
    def close_stream(self) -> UpdateServerModelRes:
        """this method should be called by the user after it finishes processing"""

    @abstractmethod
    def _get_synchronization_result(self) -> UpdateServerModelRes:
        """Possibly return some values when the client asks to close the stream"""

    def numpy_close_stream(self) -> np.ndarray:
        res = self.close_stream()
        return bytes_to_ndarray(res.result)
