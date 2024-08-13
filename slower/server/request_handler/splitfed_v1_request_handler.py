from slower.server.request_handler.request_handler import RequestHandler
from slower.server.server_model.manager.server_model_manager import ServerModelManager


class SplitFedv1ContextManager:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class SplitFedv1RequestHandler(RequestHandler):

    def __init__(self, server_model_manager: ServerModelManager):
        self.server_model_manager = server_model_manager

    def get_server_model(self, client_id):
        return SplitFedv1ContextManager(self.server_model_manager.get_server_model(client_id))
