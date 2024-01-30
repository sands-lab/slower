"""Ray-based Slower ClientProxy implementation."""

import traceback
from logging import ERROR
from typing import Optional, cast

import ray
from flwr import common
from flwr.common.logger import log


from slower.simulation.ray_transport.split_learning_actor_pool import (
    SplitLearningVirtualClientPool
)
from slower.simulation.ray_transport.typing import ClientRes, JobFn
from slower.client.client import (
    Client,
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters
)
from slower.client.typing import ClientFn
from slower.common.constants import RAY_MEMORY_LOCATION
from slower.server.server_model_segment.proxy.server_model_segment_proxy import (
    ServerModelSegmentProxy
)
from slower.server.server_model_segment.manager.server_model_segment_manager import ServerModelSegmentManager
from slower.client.proxy.client_proxy import ClientProxy


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self,
        client_fn: ClientFn,
        cid: str,
        actor_pool: SplitLearningVirtualClientPool,
        server_model_segment_manager: ServerModelSegmentManager
    ):
        super().__init__(cid)
        self.client_fn = client_fn
        self.actor_pool = actor_pool
        self.server_model_segment_manager = server_model_segment_manager

    def _submit_job(self, job_fn: JobFn, timeout: Optional[float]) -> ClientRes:
        try:
            self.actor_pool.submit_client_job(
                lambda a, c_fn, j_fn, cid: a.run.remote(c_fn, j_fn, cid),
                (self.client_fn, job_fn, self.cid),
            )
            res = self.actor_pool.get_client_result(self.cid, timeout)

        except Exception as ex:
            if self.actor_pool.num_actors == 0:
                # At this point we want to stop the simulation.
                # since no more client workloads will be executed
                log(ERROR, "ActorPool is empty!!!")
            log(ERROR, traceback.format_exc())
            log(ERROR, ex)
            raise ex

        return res

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""

        def get_parameters(client: Client) -> common.GetParametersRes:
            return maybe_call_get_parameters(
                client=client,
                get_parameters_ins=ins,
            )

        res = self._submit_job(get_parameters, timeout)

        return cast(
            common.GetParametersRes,
            res,
        )

    def fit(
        self,
        ins: common.FitIns,
        timeout: Optional[float]
    ) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        server_model_segment_proxy = self.server_model_segment_manager.get_server_model_segment_proxy(self.cid)
        def fit(client: Client) -> common.FitRes:
            # also return the server_model_segment_proxy, so that we can store it outside the
            # ray actor to the shared ray memory
            res = maybe_call_fit(
                client=client,
                fit_ins=ins,
                server_model_segment_proxy=server_model_segment_proxy,
            )
            return res, server_model_segment_proxy, client.cid

        res, server_model_segment_proxy, cid = self._submit_job(fit, timeout)
        stp_ref = ray.put(server_model_segment_proxy)

        self.actor_pool.add_object_reference_mapping(cid, stp_ref)

        # inject information
        res.metrics[RAY_MEMORY_LOCATION] = str(stp_ref.hex())

        return cast(
            common.FitRes,
            res,
        )

    def evaluate(
        self,
        ins: common.EvaluateIns,
        timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        server_model_segment_proxy = self.server_model_segment_manager.get_server_model_segment_proxy(self.cid)
        def evaluate(client: Client) -> common.EvaluateRes:
            return maybe_call_evaluate(
                client=client,
                evaluate_ins=ins,
                server_model_segment_proxy=server_model_segment_proxy,
            )

        res = self._submit_job(evaluate, timeout)

        return cast(
            common.EvaluateRes,
            res,
        )

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)
