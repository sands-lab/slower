import threading

from typing import Callable, List, Tuple

from flwr.common import FitRes

from slower.server.server_model.server_model import ServerModel
from slower.server.server_model.manager.server_model_manager import (
    ServerModelManager
)
from slower.client.proxy.client_proxy import ClientProxy
from slower.common import ServerModelFitRes
from slower.server.server_model.proxy.ray_server_model_proxy import (
    RayServerModelProxy
)
from slower.server.server_model.proxy.ray_private_server_model_proxy import (
    RayPrivateServerModelProxy
)


class GrpcServerModelManager(ServerModelManager):

    def __init__(
        self,
        init_server_model_fn: Callable[[], ServerModel],
        server_model_resources,
        common_server_model: bool
    ):
        super().__init__()
        self.common_server_model = common_server_model
        self.init_server_model_fn = lambda: init_server_model_fn().to_server_model()

        self.common_proxy = None
        if common_server_model:
            self.common_proxy = self._init_new_proxy()
            self.global_lock = threading.Lock()
        else:
            self.client_locks = {}
            self.spanned_proxies = {}

    def _init_new_proxy(self):
        proxy = RayPrivateServerModelProxy(self.init_server_model_fn())
        return proxy

    def obtain_client_lock(self, cid):
        if self.common_server_model:
            return self.global_lock

        if cid not in self.client_locks:
            self.client_locks[cid] = threading.Lock()
        return self.client_locks[cid]


    def get_server_model_proxy(self, cid) -> RayServerModelProxy:

        if self.common_proxy:
            # if all clients share the same server proxy, return it
            return self.common_proxy

        if cid in self.spanned_proxies:
            # if each client has its own proxy and the proxy for client cid is already initialized,
            # return it
            return self.spanned_proxies[cid]

        # clients have private proxies and the proxy for client cid is not initialized yet
        proxy = self._init_new_proxy()
        if cid in {"", "-1"}:
            # pylint: disable=broad-exception-raised
            raise Exception("Should not happen")
        self.configure_proxy(proxy)

        self.spanned_proxies[cid] = proxy
        return proxy

    def _get_fit_res(self, proxy: RayServerModelProxy, cid: str, num_examples: int):
        res = proxy.get_parameters(None)
        res = ServerModelFitRes(
            parameters=res.parameters,
            num_examples=num_examples,
            cid=cid
        )
        return res


    def collect_server_models(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelFitRes]:
        dataset_size_mapping = {client.cid: res.num_examples for client, res in results}
        print(dataset_size_mapping)
        if self.common_proxy:
            tot_num_examples = sum(dataset_size_mapping.values())
            return [
                self._get_fit_res(self.common_proxy, "", tot_num_examples)
            ]

        out = []
        for cid, proxy in self.spanned_proxies.items():
            print("Collecting", cid)
            res = self._get_fit_res(proxy, cid, dataset_size_mapping[cid])
            out.append(res)
        return out

    def end_round(self):
        self.spanned_proxies = {}
