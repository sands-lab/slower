from abc import ABC

from flwr.client.workload_state import WorkloadState
from flwr.client.client import Client as FlwrClient

from slower.server.server_model.proxy.server_model_proxy import ServerModelProxy


class Client(FlwrClient):

    state: WorkloadState

    def __init__(self):
        self.server_model_proxy = None

    def to_client(self):
        return self

    def set_server_model_proxy(self, server_model_proxy: ServerModelProxy):
        self.server_model_proxy = server_model_proxy
