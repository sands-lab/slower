from typing import List, Tuple, Union
from abc import ABC, abstractmethod

from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy

from slower.server.server_model.proxy.server_model_proxy import (
    ServerModelProxy
)
from slower.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes
)


class ServerModelManager(ABC):

    def __init__(self):
        self.fit_config, self.evaluation_config = None, None

    @abstractmethod
    def get_server_model_proxy(self, cid) -> ServerModelProxy:
        """Create a brand new server trainer proxy to which clients can connect"""

    @abstractmethod
    def collect_server_models(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelFitRes]:
        """Get all the necessary data for server side model aggregation"""

    def configure_proxy(self, proxy):
        if self.fit_config is not None:
            proxy.configure_fit(self.fit_config, None)
        else:
            proxy.configure_evaluate(self.evaluation_config, None)


    def set_fit_config(self, config):
        self.evaluation_config = None
        self.fit_config = config

    def set_evaluation_config(self, config):
        self.fit_config = None
        self.evaluation_config = config

    def end_round(self):
        pass
