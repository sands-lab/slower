from typing import Callable, List, Tuple

import ray
from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy

from slower.server.server_model.server_model import ServerModel
from slower.common import ServerModelFitIns, ServerModelFitRes
from slower.server.server_model.server_model import ServerModel
from slower.simulation.ray_transport.split_learning_actor_pool import SplitLearningVirtualClientPool
from slower.common.constants import RAY_MEMORY_LOCATION
from slower.server.server_model.manager.server_model_manager import ServerModelManager


class RayPrivateServerModelManager(ServerModelManager):

    def __init__(
        self,
        init_server_model_fn: Callable[[], ServerModel],
        actor_pool: SplitLearningVirtualClientPool
    ):
        super().__init__()
        self.init_server_model_fn = init_server_model_fn
        self.actor_pool = actor_pool

    def get_server_model(self, cid) -> ServerModel:
        _ = (cid,)
        assert bool(self.fit_config) != bool(self.evaluation_config)
        server_model = self.init_server_model_fn().to_server_model()

        self.configure_server_model(server_model)

        return server_model

    def collect_server_models(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelFitIns]:

        collected_server_results = []
        for client_proxy, fit_res in results:
            # get data from shared memory
            hx = bytes.fromhex(fit_res.metrics.pop(RAY_MEMORY_LOCATION))
            object_id = ray.ObjectID(hx)
            server_model: ServerModel = ray.get(object_id)
            res = server_model.get_parameters()
            res = ServerModelFitRes(
                parameters=res.parameters,
                num_examples=fit_res.num_examples,
                cid=client_proxy.cid
            )
            collected_server_results.append(res)

        # extracted the data, remove any reference so ray can free the memory
        self.actor_pool.reset_object_reference_mapping()
        return collected_server_results
