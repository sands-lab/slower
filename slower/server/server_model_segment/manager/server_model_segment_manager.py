from typing import List, Tuple, Union
from abc import ABC, abstractmethod

from flwr.common import FitRes

from slower.server.server_model_segment.proxy.server_model_segment_proxy import (
    ServerModelSegmentProxy
)
from slower.common import (
    ServerModelSegmentEvaluateIns,
    ServerModelSegmentFitIns,
    ServerModelSegmentFitRes
)
from slower.client.proxy.client_proxy import ClientProxy


class ServerModelSegmentManager(ABC):

    @abstractmethod
    def get_server_model_segment_proxy(self, cid) -> ServerModelSegmentProxy:
        """Create a brand new server trainer proxy to which clients can connect"""

    @abstractmethod
    def collect_server_model_segments(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelSegmentFitRes]:
        """Get all the necessary data for server side model aggregation"""

    # pylint: disable=unused-argument
    def init_server_model_segment_proxies(
        self,
        server_round: int,
        cids: int,
        server_model_segment_config: Union[ServerModelSegmentFitIns, ServerModelSegmentEvaluateIns]
    ) -> List[ServerModelSegmentProxy]:
        if isinstance(server_model_segment_config, ServerModelSegmentFitIns):
            # pylint: disable=unnecessary-lambda-assignment
            config_fn = lambda proxy: proxy.configure_fit(server_model_segment_config, None)
        else:
            assert isinstance(server_model_segment_config, ServerModelSegmentEvaluateIns)
            # pylint: disable=unnecessary-lambda-assignment
            config_fn = lambda proxy: proxy.configure_evaluate(server_model_segment_config, None)

        proxies = []
        for cid in cids:
            proxy = self.get_server_model_segment_proxy(cid)
            config_fn(proxy)
            proxies.append(proxy)

        return proxies
