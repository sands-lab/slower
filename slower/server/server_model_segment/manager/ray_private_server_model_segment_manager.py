from typing import Callable, List, Tuple

import ray
from flwr.common import FitRes

from slower.server.server_model_segment.proxy.server_model_segment_proxy import (
    ServerModelSegmentProxy
)
from slower.common import (
    ServerModelSegmentFitIns,
    ServerModelSegmentFitRes
)
from slower.client.proxy.client_proxy import ClientProxy
from slower.server.server_model_segment.server_model_segment import ServerModelSegment
from slower.simulation.ray_transport.split_learning_actor_pool import SplitLearningVirtualClientPool
from slower.server.server_model_segment.proxy.ray_private_server_model_segment_proxy import (
    RayPrivateServerModelSegmentProxy
)
from slower.common.constants import RAY_MEMORY_LOCATION
from slower.server.server_model_segment.manager.server_model_segment_manager import (
    ServerModelSegmentManager
)


class RayPrivateServerModelSegmentManager(ServerModelSegmentManager):

    def __init__(
        self,
        init_server_model_segment_fn: Callable[[], ServerModelSegment],
        actor_pool: SplitLearningVirtualClientPool
    ):
        super().__init__()
        self.init_server_model_segment_fn = init_server_model_segment_fn
        self.actor_pool = actor_pool

    def get_server_model_segment_proxy(self, cid) -> ServerModelSegmentProxy:
        assert bool(self.fit_config) != bool(self.evaluation_config)
        server_model_segment = self.init_server_model_segment_fn().to_server_model_segment()
        proxy = RayPrivateServerModelSegmentProxy(server_model_segment)

        if self.fit_config is not None:
            proxy.configure_fit(self.fit_config, None)
        else:
            proxy.configure_evaluate(self.evaluation_config, None)

        return proxy

    def collect_server_model_segments(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelSegmentFitIns]:

        collected_server_results = []
        for client_proxy, fit_res in results:
            # get data from shared memory
            hx = bytes.fromhex(fit_res.metrics.pop(RAY_MEMORY_LOCATION))
            object_id = ray.ObjectID(hx)
            server_model_segment: ServerModelSegment = ray.get(object_id)
            res = server_model_segment.get_parameters(None)
            res = ServerModelSegmentFitRes(
                parameters=res.parameters,
                num_examples=fit_res.num_examples,
                cid=client_proxy.cid
            )
            collected_server_results.append(res)

        # extracted the data, remove any reference so ray can free the memory
        self.actor_pool.reset_object_reference_mapping()
        return collected_server_results
