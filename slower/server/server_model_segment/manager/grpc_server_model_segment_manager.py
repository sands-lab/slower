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
from slower.server.server_model_segment.proxy.ray_server_model_segment_proxy import (
    RayServerModelSegmentProxy
)
from slower.simulation.ray_transport.server_model_segment_actor import VirtualServerSegmentModelActor
from slower.server.server_model_segment.proxy.ray_private_server_model_segment_proxy import RayPrivateServerModelSegmentProxy


class GrpcServerModelSegmentManager(ServerModelSegmentManager):

    def __init__(
        self,
        init_server_model_segment_fn: Callable[[], ServerModelSegment],
        server_model_segment_resources,
        common_server_model_segment: bool
    ):
        self.server_model_segment = init_server_model_segment_fn()
        self.server_model_segment_resources = server_model_segment_resources

        self.common_proxy = None
        if common_server_model_segment:
            self.common_proxy = self._init_new_proxy()

        self.spanned_proxies = {}

    def _init_new_proxy(self):
        # pylint: disable=no-member
        # server_model_segment_actor = VirtualServerSegmentModelActor\
        #     .options(**self.server_model_segment_resources)\
        #     .remote(self.server_model_segment)
        # proxy = RayServerModelSegmentProxy(
        #     server_model_segment_actor=server_model_segment_actor
        # )
        proxy = RayPrivateServerModelSegmentProxy(self.server_model_segment)
        return proxy


    def get_server_model_segment_proxy(self, cid) -> RayServerModelSegmentProxy:
        if self.common_proxy:
            # if all clients share the same server proxy, return it
            return self.common_proxy

        if cid in self.spanned_proxies:
            # if each client has its own proxy and the proxy for client cid is already initialized,
            # return it
            return self.spanned_proxies[cid]

        # clients have private proxies and the proxy for client cid is not initialized yet
        proxy = self._init_new_proxy()
        if cid == "" or cid == "-1":
            raise Exception("Should not happen")
        self.spanned_proxies[cid] = proxy
        return proxy

    def _get_fit_res(self, proxy: RayServerModelSegmentProxy, cid: str, num_examples: int):
        res = proxy.get_parameters(None)
        res = ServerModelSegmentFitRes(
            parameters=res.parameters,
            num_examples=num_examples,
            cid=cid
        )
        return res


    def collect_server_model_segments(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelSegmentFitRes]:
        dataset_size_mapping = {client.cid: res.num_examples for client, res in results}
        if self.common_proxy:
            tot_num_examples = sum(dataset_size_mapping.values())
            return [
                self._get_fit_res(self.common_proxy, "", tot_num_examples)
            ]

        out = []
        for cid, proxy in self.spanned_proxies.items():
            res = self._get_fit_res(proxy, cid, dataset_size_mapping[cid])
            out.append(res)
        self.spanned_proxies = {}  # reset it so that ray can do garbage collection
        return out
