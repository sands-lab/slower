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
    ):
        super().__init__()
        _ = (server_model_resources,)  # this should be handled in the future
        self.init_server_model_fn = lambda: init_server_model_fn().to_server_model()

        self.client_locks = {}
        self.spanned_server_models = {}

    def get_server_model(self, cid) -> ServerModel:

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
        if None in self.spanned_server_models:
            assert len(self.spanned_server_models) == 1, \
                "None is allowed as key only when all clients share the same model"
            tot_num_examples = sum(dataset_size_mapping.values())
            return [
                self._get_fit_res(self.spanned_server_models[None], "", tot_num_examples)
            ]

        out = []
        for cid, proxy in self.spanned_server_models.items():
            res = self._get_fit_res(proxy, cid, dataset_size_mapping[cid])
            out.append(res)
        return out

    def end_round(self):
        self.spanned_server_models = {}
