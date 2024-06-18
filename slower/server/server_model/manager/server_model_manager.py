import traceback
from typing import List, Tuple
from abc import ABC, abstractmethod
from logging import ERROR

from flwr.common import FitRes
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from slower.server.server_model.proxy.server_model_proxy import ServerModelProxy
from slower.common import ServerModelFitRes


class ServerModelManager(ABC):

    def __init__(self):
        self.fit_config, self.evaluation_config = None, None

    @abstractmethod
    def get_server_model(self, cid) -> ServerModelProxy:
        """Create a brand new server trainer proxy to which clients can connect"""

    @abstractmethod
    def collect_server_models(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelFitRes]:
        """Get all the necessary data for server side model aggregation"""

    def configure_server_model(self, server_model):
        try:
            if self.fit_config is not None:
                server_model.configure_fit(self.fit_config)
            else:
                server_model.configure_evaluate(self.evaluation_config)
        except Exception as e:
            log(ERROR, e)
            log(ERROR, traceback.format_exc())
            raise e

    def set_fit_config(self, config):
        self.evaluation_config = None
        self.fit_config = config

    def set_evaluation_config(self, config):
        self.fit_config = None
        self.evaluation_config = config

    def end_round(self):
        pass
