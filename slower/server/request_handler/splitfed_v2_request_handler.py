import threading

from slower.server.request_handler.request_handler import RequestHandler


class SplitFedv2ContextManager:
    def __init__(self, lock, server_model_manager):
        self.lock = lock
        self.server_model_manager = server_model_manager

    def __enter__(self):
        self.lock.acquire()
        return self.server_model_manager.get_server_model(None)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()


class SplitFedv2RequestHandler(RequestHandler):

    def __init__(self, server_model_manager):
        self.server_model_manager = server_model_manager
        self.lock = threading.RLock()

    def get_server_model(self, client_id):
        return SplitFedv2ContextManager(self.lock, self.server_model_manager)
