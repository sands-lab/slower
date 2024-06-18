from queue import SimpleQueue
from typing import Iterator
import threading

from slower.server.server_model.server_model import ServerModel
from slower.common import ControlCode, BatchData
from slower.server.server_model.proxy.server_model_proxy import ServerModelProxy


class RayPrivateServerModelProxy(ServerModelProxy):
    """Slower server model proxy which which acts as a POPO inside the ray actor (the client)."""

    def __init__(
        self,
        server_model: ServerModel,
        request_queue_in_separate_thread: bool = True
    ):
        super().__init__()
        self.server_model = server_model
        self.request_queue = None
        self.server_request_thread = None
        self.request_queue_in_separate_thread = request_queue_in_separate_thread

    def _blocking_request(self, method, batch_data, timeout):
        _ = (timeout,)
        res = getattr(self.server_model, method)(batch_data=batch_data)
        return res

    def _streaming_request(self, method, batch_data):

        if self.request_queue is not None:
            self.request_queue.put((method, batch_data))
        else:
            self._blocking_request(method=method, batch_data=batch_data, timeout=None)

    def _is_stream_initialized(self):
        return not self.request_queue_in_separate_thread or self.request_queue is not None

    def _initialize_stream(self):

        # start a new thread that will handle the requests
        def _iterate(server_proxy: ServerModelProxy, iterator: Iterator):
            for method, batch in iterator:
                if batch.control_code == ControlCode.DO_CLOSE_STREAM:
                    break
                server_proxy._blocking_request(method, batch, None)

        self.request_queue = SimpleQueue()

        queue_iterator = iter(self.request_queue.get, None)
        self.server_request_thread = threading.Thread(
            target=_iterate,
            args=(self, queue_iterator,)
        )
        self.server_request_thread.start()

    def close_stream(self) -> BatchData:
        if self.request_queue is not None:
            ins = BatchData(
                data={},
                control_code=ControlCode.DO_CLOSE_STREAM
            )
            self.request_queue.put(("", ins))
            self.server_request_thread.join()
            res = self.server_model.get_synchronization_result()
            qsize = self.request_queue.qsize()
        else:
            # trivially, return that the stream was closed ok, simply because there
            # was no stream in the first place
            res = self.server_model.get_synchronization_result()
            qsize = 0

        self.request_queue = None
        self.server_request_thread = None

        if qsize > 0:
            # pylint: disable=broad-exception-raised
            raise Exception(f"Request queue is not empty!! Size: {qsize}")
        return res

    def get_pending_batches_count(self) -> int:
        return self.request_queue.qsize()
