import threading

from typing import Callable, List, Tuple

from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy

from slower.server.server_model.server_model import ServerModel
from slower.server.server_model.manager.server_model_manager import ServerModelManager
from slower.common import ServerModelFitRes


class GrpcServerModelManager(ServerModelManager):

    def __init__(
        self,
        init_server_model_fn: Callable[[], ServerModel],
        server_model_resources,
        common_server_model: bool
    ):
        super().__init__()
        _ = (server_model_resources,)  # this should be handled in the future
        self.common_server_model = common_server_model
        self.init_server_model_fn = lambda: init_server_model_fn().to_server_model()

        assert not common_server_model  # this functionality is not yet fully supported
        self.common_server_model = None
        if common_server_model:
            self.common_server_model = self.init_server_model_fn()
            self.global_lock = threading.Lock()
        else:
            self.client_locks = {}
            self.spanned_server_models = {}

    def get_server_model(self, cid) -> ServerModel:

        if self.common_server_model:
            # if all clients share the same server proxy, return it
            return self.common_server_model

        if cid in self.spanned_server_models:
            # if each client has its own model and the model for client cid is already initialized,
            # return it
            return self.spanned_server_models[cid]

        # clients have private proxies and the proxy for client cid is not initialized yet
        sm = self.init_server_model_fn()
        if cid in {"", "-1"}:
            # pylint: disable=broad-exception-raised
            raise Exception("Should not happen")
        self.configure_server_model(sm)

        self.spanned_server_models[cid] = sm
        return sm

    def _get_fit_res(self, server_model: ServerModel, cid: str, num_examples: int):
        res = server_model.get_parameters()
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
        if self.common_server_model:
            tot_num_examples = sum(dataset_size_mapping.values())
            return [
                self._get_fit_res(self.common_proxy, "", tot_num_examples)
            ]

        out = []
        for cid, proxy in self.spanned_server_models.items():
            res = self._get_fit_res(proxy, cid, dataset_size_mapping[cid])
            out.append(res)
        return out

    def end_round(self):
        self.spanned_server_models = {}
