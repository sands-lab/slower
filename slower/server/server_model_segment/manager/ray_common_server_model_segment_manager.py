from typing import Callable, List, Tuple

from flwr.common import FitRes

from slower.server.server_model_segment.server_model_segment import ServerModelSegment
from slower.server.server_model_segment.manager.server_model_segment_manager import (
    ServerModelSegmentManager
)
from slower.simulation.ray_transport.server_model_segment_actor import (
    VirtualServerSegmentModelActor
)
from slower.client.proxy.client_proxy import ClientProxy
from slower.common import ServerModelSegmentFitRes
from slower.server.server_model_segment.proxy.ray_common_server_model_segment_proxy import (
    RayCommonServerModelSegmentProxy
)


class RayCommonServerModelSegmentManager(ServerModelSegmentManager):

    def __init__(
        self,
        init_server_model_segment_fn: Callable[[], ServerModelSegment],
        server_model_segment_resources
    ):
        server_model_segment = init_server_model_segment_fn()
        # pylint: disable=no-member
        server_model_segment_actor = VirtualServerSegmentModelActor\
            .options(**server_model_segment_resources)\
            .remote(server_model_segment)
        self.server_model_segment_proxy = \
            RayCommonServerModelSegmentProxy(server_model_segment_actor)

    def get_server_model_segment_proxy(self) -> RayCommonServerModelSegmentProxy:
        return self.server_model_segment_proxy

    #pylint: disable=unused-argument
    def collect_server_model_segments(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelSegmentFitRes]:
        res = self.server_model_segment_proxy.get_parameters(None)
        server_fit_res = ServerModelSegmentFitRes(
            parameters=res.parameters,
            num_examples=1,
            cid=""
        )
        return [server_fit_res]
