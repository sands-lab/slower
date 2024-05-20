from typing import Optional, Iterator, Dict, Union, List
from abc import ABC, abstractmethod

import numpy as np
from flwr.common import GetParametersRes

from slower.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ControlCode,
    BatchData
)
from slower.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict


NumpyBatchData = Dict[str, Union[np.ndarray, List[np.ndarray]]]


class ServerModelProxy(ABC):

    @abstractmethod
    def serve_prediction_request(
        self,
        batch_data: BatchData,
        timeout: Optional[float]
    ) -> BatchData:
        """Compute the final predictions"""

    @abstractmethod
    def serve_gradient_update_request(
        self,
        batch_data: BatchData,
        timeout: Optional[float]
    ) -> BatchData:
        """Compute the loss, and backpropagate it and send to the client the gradient info"""

    @abstractmethod
    def get_parameters(
        self,
        timeout: Optional[float]
    ) -> GetParametersRes:
        """Return the weights of the server model"""

    @abstractmethod
    def update_server_model(self, batch_data: BatchData):
        """Receives a single batch and sends it to the server for being processed"""

    @abstractmethod
    def u_forward(self, batch_data: BatchData) -> BatchData:
        """Invoke the forward pass on the server-side model"""

    @abstractmethod
    def u_backward(self, batch_data: BatchData, blocking=True) -> BatchData:
        """Invoke the backward pass on the server. If blocking is set to true, the client will wait
        until it gets a response from the server, otherwise the client will just send the data"""

    @abstractmethod
    def close_stream(self) -> BatchData:
        """this method should be called by the user after it finishes processing"""

    @abstractmethod
    def _get_synchronization_result(self) -> BatchData:
        """Possibly return some values when the client asks to close the stream"""

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
        batch_data: NumpyBatchData,
        timeout: Optional[float] = None
    ) -> NumpyBatchData:
        ins = BatchData(
            data=ndarray_dict_to_bytes(batch_data),
            control_code=ControlCode.OK
        )
        res = self.serve_prediction_request(
            batch_data=ins,
            timeout=timeout
        )
        return bytes_to_ndarray_dict(res.data)


    def numpy_serve_gradient_update_request(
        self,
        batch_data: NumpyBatchData,
        timeout: Optional[float] = None
    ) -> NumpyBatchData:
        ins= BatchData(
            data=ndarray_dict_to_bytes(batch_data),
            control_code=ControlCode.OK
        )
        res = self.serve_gradient_update_request(
            batch_data=ins,
            timeout=timeout,
        )
        return bytes_to_ndarray_dict(res.data)


    def numpy_update_server_model(
        self,
        batch_data: NumpyBatchData
    ) -> None:
        ins = BatchData(
            data=ndarray_dict_to_bytes(batch_data),
            control_code=ControlCode.OK,
        )
        self.update_server_model(ins)

    def _update_server_model_requests(
        self, batches_iterator: Iterator[BatchData]
    ) -> None:

        for batch in batches_iterator:
            if batch.control_code == ControlCode.DO_CLOSE_STREAM:
                break
            self.serve_gradient_update_request(batch, None)

    def numpy_u_forward(
        self,
        batch_data: NumpyBatchData
    ) -> NumpyBatchData:
        ins = BatchData(
            data=ndarray_dict_to_bytes(batch_data),
            control_code=ControlCode.OK
        )
        res = self.u_forward(batch_data=ins)
        return bytes_to_ndarray_dict(res.data)

    def numpy_u_backward(
        self,
        batch_data: NumpyBatchData,
        blocking=True
    ) -> BatchData:
        ins = BatchData(
            data=ndarray_dict_to_bytes(batch_data),
            control_code=ControlCode.OK
        )
        res = self.u_backward(batch_data=ins, blocking=blocking)
        return bytes_to_ndarray_dict(res.data)

    def numpy_close_stream(self) -> np.ndarray:
        res = self.close_stream()
        return bytes_to_ndarray_dict(res.data)

    def get_pending_batches_count(self) -> int:
        """In the streaming API, returns the number of batches that are unprocessed (i.e., how much is the client ahead of the server)

        Returns
        -------
        int
            number of pending batches
        """
