from flwr.client.client import Client as FlwrClient

from slwr.server.server_model.proxy.server_model_proxy import ServerModelProxy


class Client(FlwrClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_model_proxy = None

    def set_server_model_proxy(
        self,
        server_model_proxy: ServerModelProxy
    ):
        self.server_model_proxy = server_model_proxy
