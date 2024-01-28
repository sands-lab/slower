from abc import ABC
from logging import WARNING

import ray
from flwr.common.logger import log

from slower.server.server_model_segment.server_model_segment import ServerModelSegment


@ray.remote
class VirtualServerSegmentModelActor(ABC):
    """Abstract base class for VirtualClientEngine Actors."""

    def __init__(self, server_model_segment: ServerModelSegment) -> None:
        super().__init__()
        self.server_model_segment: ServerModelSegment = server_model_segment

    def terminate(self) -> None:
        """Manually terminate Actor object."""
        log(WARNING, "Manually terminating %s}", self.__class__.__name__)
        ray.actor.exit_actor()

    def get_parameters(self, *args, **kwargs):
        return self.server_model_segment.get_parameters(*args, **kwargs)

    def configure_fit(self, *args, **kwargs):
        return self.server_model_segment.configure_fit(*args, **kwargs)

    def configure_evaluate(self, *args, **kwargs):
        return self.server_model_segment.configure_evaluate(*args, **kwargs)

    def serve_prediction_request(self, *args, **kwargs):
        return self.server_model_segment.serve_prediction_request(*args, **kwargs)

    def serve_gradient_update_request(self, *args, **kwargs):
        return self.server_model_segment.serve_gradient_update_request(*args, **kwargs)

    def get_server_model_segment(self):
        return self.server_model_segment
