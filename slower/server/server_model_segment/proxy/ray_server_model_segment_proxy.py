import traceback
from logging import ERROR
from typing import Optional

import ray
from flwr.common.logger import log
from flwr.common.typing import GetParametersRes

from slower.simulation.ray_transport.server_model_segment_actor import (
    VirtualServerSegmentModelActor
)
from slower.simulation.ray_transport.typing import ServerRes
from slower.common import (
    ServerModelSegmentFitIns,
    ServerModelSegmentEvaluateIns,
    BatchPredictionIns,
    BatchPredictionRes,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes
)
from slower.server.server_model_segment.proxy.server_model_segment_proxy import (
    ServerModelSegmentProxy
)


class RayServerModelSegmentProxy(ServerModelSegmentProxy):
    """Flower client proxy which delegates work using Ray."""


    def __init__(
        self,
        server_model_segment_actor: VirtualServerSegmentModelActor
    ):
        super().__init__()
        self.server_model_segment_actor = server_model_segment_actor

    def _request(self, ref, timeout: Optional[float]) -> ServerRes:
        try:

            result = ray.get(ref, timeout=timeout)

        except Exception as ex:
            log(ERROR, traceback.format_exc())
            log(ERROR, ex)
            raise ex

        return result

    def get_parameters(
        self, timeout: Optional[float]
    ) -> GetParametersRes:
        """Return the current local model parameters."""
        ref = self.server_model_segment_actor.get_parameters.remote()
        res = self._request(ref, timeout)

        return res

    def configure_fit(
        self, ins: ServerModelSegmentFitIns, timeout: Optional[float]
    ) -> bool:
        """Return the current local model parameters."""
        ref = self.server_model_segment_actor.configure_fit.remote(ins)
        res = self._request(ref, timeout)

        return res

    def configure_evaluate(
        self, ins: ServerModelSegmentEvaluateIns, timeout: Optional[float]
    ) -> bool:
        """Return the current local model parameters."""
        ref = self.server_model_segment_actor.configure_evaluate.remote(ins)
        res = self._request(ref, timeout)

        return res

    def serve_prediction_request(
        self,
        batch_data: BatchPredictionIns,
        timeout: Optional[float]
    ) -> BatchPredictionRes:
        ref = self.server_model_segment_actor.serve_prediction_request.remote(batch_data)
        res = self._request(ref, timeout)
        return res

    def serve_gradient_update_request(
        self,
        batch_data: GradientDescentDataBatchIns,
        timeout: Optional[float]
    ) -> GradientDescentDataBatchRes:
        """Compute the loss and backpropagate it to the client"""
        ref = self.server_model_segment_actor.serve_gradient_update_request.remote(batch_data)
        res = self._request(ref, timeout)
        return res
