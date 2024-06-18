from queue import SimpleQueue

from slower.common import BatchData
from slower.common.serde import (
    control_code_to_proto,
    control_code_from_proto,
    from_grpc_format,
    to_grpc_format
)
from slower.server.server_model.proxy.server_model_proxy import ServerModelProxy
from slower.proto import server_model_pb2_grpc
from slower.proto import server_model_pb2


class ClientException(Exception):
    """Exception indicating that the client invoked a method it should not have. For instance,
    server model configuration should be taken care of by the server, and any attempt
    at doing so by the client will result in an exception. The client may only use this proxy
    for computing predictions and for requesting gradient updates."""


# pylint: disable=no-member
class GrpcServerModelProxy(ServerModelProxy):

    def __init__(self, stub: server_model_pb2_grpc.ServerModelStub, cid: str) -> None:
        super().__init__()
        self.stub = stub
        self.cid = cid
        self.request_queue, self.request_future = None, None

    def _blocking_request(self, method, batch_data, timeout):
        ins = server_model_pb2.BatchData(
            method=method,
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )
        res = self.stub.BlockingRequest(ins)
        return BatchData(
            data=from_grpc_format(res.data),
            control_code=control_code_from_proto(res.control_code)
        )

    def _streaming_request(self, method, batch_data):
        ins = server_model_pb2.BatchData(
            method=method,
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )
        self.request_queue.put(ins)

    def _is_stream_initialized(self):
        return self.request_queue is not None

    def _initialize_stream(self):
        self.request_queue = SimpleQueue()
        queue_iterator = iter(self.request_queue.get, None)
        self.request_future = self.stub.StreamingRequests.future(queue_iterator)

    def close_stream(self) -> BatchData:
        if self.request_queue is None:
            raise ClientException("Trying to close_stream but stream is not open!")
        self.request_queue.put(server_model_pb2.BatchData(
            method="",
            data={},
            control_code=server_model_pb2.ControlCode.DO_CLOSE_STREAM
        ))
        result = self.request_future.result()
        self.request_queue, self.request_future = None, None
        return BatchData(
            data=from_grpc_format(result.data),
            control_code=control_code_from_proto(result.control_code),
        )

    def get_pending_batches_count(self) -> int:
        if self.request_queue is None:
            raise ClientException("Stream is not initialized!")
        return self.request_queue.qsize()
