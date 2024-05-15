from abc import ABC
from logging import WARNING

import ray
from flwr.common.logger import log

from slower.server.server_model.server_model import ServerModel


@ray.remote
class VirtualServerModelActor(ABC):
    """Abstract base class for VirtualClientEngine Actors."""

    def __init__(self, server_model: ServerModel) -> None:
        super().__init__()
        self.server_model: ServerModel = server_model

    def terminate(self) -> None:
        """Manually terminate Actor object."""
        log(WARNING, "Manually terminating %s}", self.__class__.__name__)
        ray.actor.exit_actor()

    def get_parameters(self, *args, **kwargs):
        return self.server_model.get_parameters(*args, **kwargs)

    def configure_fit(self, *args, **kwargs):
        return self.server_model.configure_fit(*args, **kwargs)

    def configure_evaluate(self, *args, **kwargs):
        return self.server_model.configure_evaluate(*args, **kwargs)

    def serve_prediction_request(self, *args, **kwargs):
        return self.server_model.serve_prediction_request(*args, **kwargs)

    def serve_gradient_update_request(self, *args, **kwargs):
        return self.server_model.serve_gradient_update_request(*args, **kwargs)

    def get_server_model(self):
        return self.server_model
