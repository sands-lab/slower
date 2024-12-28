from typing import Callable, List, Union

from slwr.server.server_model.server_model import ServerModel
from slwr.server.server_model.manager import ServerModelManager
from slwr.common import ServerModelFitRes, ServerModelEvaluateIns, ServerModelFitIns


class GrpcServerModelManager(ServerModelManager):

    def __init__(
        self,
        init_server_model_fn: Callable[[], ServerModel],
    ) -> None:
        super().__init__()
        self.init_server_model_fn = lambda: init_server_model_fn().to_server_model()
        self.client_locks = {}
        self.spanned_server_models = {}

    def get_server_model(self, sid) -> ServerModel:
        return self.spanned_server_models[sid]

    def collect_server_fit_results(self) -> List[ServerModelFitRes]:
        out = []
        for sm in self.spanned_server_models.values():
            res = sm.get_fit_result()
            out.append(res)
        return out

    def end_round(self):
        self.spanned_server_models = {}

    def initialize_server_models(
        self,
        configs: List[Union[ServerModelEvaluateIns, ServerModelFitIns]]
    ):
        is_training = isinstance(configs[0], ServerModelFitIns)
        for config in configs:
            sm = self.init_server_model_fn()

            if is_training:
                sm.configure_fit(config)
            else:
                sm.configure_evaluate(config)

            self.spanned_server_models[config.sid] = sm

    def get_server_model_ids(self):
        return list(self.spanned_server_models.keys())
