from typing import Callable, List, Tuple

from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy

from slower.server.server_model.server_model import ServerModel
from slower.server.server_model.manager.server_model_manager import (
    ServerModelManager
)
from slower.simulation.ray_transport.server_model_actor import VirtualServerModelActor
from slower.common import ServerModelFitRes
from slower.server.server_model.proxy.ray_server_model_proxy import RayServerModelProxy


class RayCommonServerModelManager(ServerModelManager):

    def __init__(
        self,
        init_server_model_fn: Callable[[], ServerModel],
        server_model_resources
    ):
        super().__init__()
        server_model = init_server_model_fn().to_server_model()
        # pylint: disable=no-member
        server_model_actor = VirtualServerModelActor\
            .options(**server_model_resources)\
            .remote(server_model)
        self.server_model_proxy = \
            RayServerModelProxy(server_model_actor)

    def get_server_model_proxy(self, cid) -> RayServerModelProxy:
        return self.server_model_proxy

    #pylint: disable=unused-argument
    def collect_server_models(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[ServerModelFitRes]:
        res = self.server_model_proxy.get_parameters(None)
        server_fit_res = ServerModelFitRes(
            parameters=res.parameters,
            num_examples=1,
            cid=""
        )
        return [server_fit_res]
