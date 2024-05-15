from queue import SimpleQueue
from typing import Optional
import threading

from flwr.common import GetParametersRes

from slower.server.server_model.server_model import ServerModel
from slower.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    BatchPredictionIns,
    BatchPredictionRes,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    ControlCode,
    UpdateServerModelRes,
    DataBatchBackward,
    DataBatchForward
)
from slower.server.server_model.proxy.server_model_proxy import ServerModelProxy


class RayPrivateServerModelProxy(ServerModelProxy):
    """Slower server model proxy which which acts as a POPO inside the ray actor (the client)."""

    def __init__(
        self,
        server_model: ServerModel,
        request_queue_in_separate_thread: bool = True
    ):
        super().__init__()
        self.server_model = server_model
        self.request_queue = None
        self.server_request_thread = None
        self.request_queue_in_separate_thread = request_queue_in_separate_thread

    def get_parameters(
        self,
        timeout: Optional[float]
    ) -> GetParametersRes:
        return self.server_model.get_parameters()

    def configure_fit(
        self, ins: ServerModelFitIns, timeout: Optional[float]
    ):
        self.server_model.configure_fit(ins)

    def configure_evaluate(
        self, ins: ServerModelEvaluateIns, timeout: Optional[float]
    ):
        self.server_model.configure_evaluate(ins)

    def serve_prediction_request(
        self,
        batch_data: BatchPredictionIns,
        timeout: Optional[float]
    ) -> BatchPredictionRes:
        return self.server_model.serve_prediction_request(batch_data)

    def serve_gradient_update_request(
        self,
        batch_data: GradientDescentDataBatchIns,
        timeout: Optional[float]
    ) -> GradientDescentDataBatchRes:
        return self.server_model.serve_gradient_update_request(batch_data)

    def update_server_model(self, batch_data: GradientDescentDataBatchIns):

        if self.request_queue is None and self.request_queue_in_separate_thread:
            # start a new thread that will handle the requests
            self.request_queue = SimpleQueue()

            queue_iterator = iter(self.request_queue.get, None)
            self.server_request_thread = threading.Thread(
                target=self._update_server_model_requests,
                args=(queue_iterator,)
            )
            self.server_request_thread.start()

        if self.request_queue is not None:
            self.request_queue.put(batch_data)
        else:
            self.serve_gradient_update_request(batch_data=batch_data, timeout=None)

    def u_forward(self, batch: DataBatchForward):
        return self.server_model.u_forward(batch)

    def u_backward(self, batch_gradient: DataBatchBackward, blocking=True):
        # blocking is ignored in ray simulations
        _ = (blocking, )
        return self.server_model.u_backward(batch_gradient)

    def close_stream(self) -> UpdateServerModelRes:
        if self.request_queue is not None:
            ins = GradientDescentDataBatchIns(
                embeddings=b"",
                labels=b"",
                control_code=ControlCode.DO_CLOSE_STREAM
            )
            self.request_queue.put(ins)
            self.server_request_thread.join()
            res = self._get_synchronization_result()
            qsize = self.request_queue.qsize()
        else:
            # trivially, return that the stream was closed ok, simply because there
            # was no stream in the first place
            res = self._get_synchronization_result()
            qsize = 0

        self.request_queue = None
        self.server_request_thread = None

        if qsize > 0:
            # pylint: disable=broad-exception-raised
            raise Exception(f"Request queue is not empty!! Size: {qsize}")
        return res

    def _get_synchronization_result(self):
        return self.server_model.get_synchronization_result()
