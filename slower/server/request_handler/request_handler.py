from abc import ABC, abstractmethod
from typing import ContextManager


class RequestHandler(ABC):

    @abstractmethod
    def __init__(self, server_model_manager):
        pass

    @abstractmethod
    def get_server_model(self, client_id) -> ContextManager:
        pass
