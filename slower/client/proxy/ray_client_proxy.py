"""Ray-based Slower ClientProxy implementation."""

from typing import Optional, cast

import ray
from flwr import common
from flwr.client.client import maybe_call_fit, maybe_call_evaluate
from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy

from slower.client.client import Client
from slower.common.constants import RAY_MEMORY_LOCATION
from slower.server.server_model.manager.server_model_manager import ServerModelManager


class RayClientProxy(RayActorClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self,
        server_model_manager: ServerModelManager,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_model_manager = server_model_manager

    def fit(
        self,
        ins: common.FitIns,
        timeout: Optional[float]
    ) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        server_model_proxy = \
            self.server_model_manager.get_server_model_proxy(self.cid)
        def fit(client: Client) -> common.FitRes:
            # also return the server_model_proxy, so that we can store it outside the
            # ray actor to the shared ray memory
            client.set_server_model_proxy(server_model_proxy=server_model_proxy)
            res = maybe_call_fit(
                client=client,
                fit_ins=ins,
            )
            return res, server_model_proxy, client.cid

        res, server_model_proxy, cid = self._submit_job(fit, timeout)
        stp_ref = ray.put(server_model_proxy)

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
        server_model_proxy = \
            self.server_model_manager.get_server_model_proxy(self.cid)
        def evaluate(client: Client) -> common.EvaluateRes:
            client.set_server_model_proxy(server_model_proxy=server_model_proxy)
            return maybe_call_evaluate(
                client=client,
                evaluate_ins=ins,
            )

        res = self._submit_job(evaluate, timeout)

        return cast(
            common.EvaluateRes,
            res,
        )
