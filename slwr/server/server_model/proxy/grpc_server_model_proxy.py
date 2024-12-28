from queue import Queue
from typing import Optional
from logging import DEBUG

from flwr.common import log

from slwr.common import BatchData
from slwr.common.serde import (
    control_code_to_proto,
    control_code_from_proto,
    from_grpc_format,
    to_grpc_format,
    ControlCode,
)
from slwr.server.server_model.proxy.server_model_proxy import ServerModelProxy
from slwr.proto import server_model_pb2_grpc
from slwr.proto import server_model_pb2


class ClientException(Exception):
    """Exception indicating that the client invoked a method it should not have. For instance,
    server model configuration should be taken care of by the server, and any attempt
    at doing so by the client will result in an exception. The client may only use this proxy
    for computing predictions and for requesting gradient updates."""


class FutureResponse:
    def __init__(self, response_future):
        self.response_future = response_future

    def get_response(self):
        return self.response_future()



# pylint: disable=no-member
class GrpcServerModelProxy(ServerModelProxy):

    """
    In all server invokations the timeout parameter is currently ignored. This is because there is
    currently no way to set a timeout for streaming requests. See:
    - https://github.com/grpc/grpc/issues/20562
    """
    def __init__(
        self,
        *,
        stub: server_model_pb2_grpc.ServerModelStub,
        cid: str
    ) -> None:
        super().__init__(cid=cid)
        log(DEBUG, "Initializing proxy for CID %s", cid)
        self.stub = stub
        self.request_queue = None

    def _initialize_stream(self):
        self.request_queue = Queue(maxsize=20)
        self.response_stream = self.stub.StreamRequest(iter(self.request_queue.get, None))
        self.request_queue.put(server_model_pb2.BatchData(
            control_code=control_code_to_proto(ControlCode.INIT_STREAM),
            cid=self.cid,
        ))

    def _blocking_request(
        self,
        method: str,
        batch_data: BatchData,
        _streams_: bool,
        _timeout_: Optional[float],
    ) -> BatchData:
        """Issue a blocking request to the server over gRPC

        Parameters
        ----------
        method : str
            name of the method to be invoked
        batch_data : BatchData
            data to be sent to the server
        _timeout_ : Optional[float]
            optionally pass the maximum time allowed for the server to return a response

        Returns
        -------
        BatchData
            data returned by the server
        """
        if _streams_:
            if self.request_queue is None:
                self._initialize_stream()

            ins = server_model_pb2.BatchData(
                method=method,
                data=to_grpc_format(batch_data.data),
                control_code=control_code_to_proto(batch_data.control_code),
            )
            self.request_queue.put(ins)
            res = next(self.response_stream)
        else:

            res = self.stub.UnaryRequest(
                server_model_pb2.BatchData(
                    method=method,
                    data=to_grpc_format(batch_data.data),
                    control_code=control_code_to_proto(batch_data.control_code),
                    cid=self.cid,
                )
            )
        return BatchData(
            data=from_grpc_format(res.data),
            control_code=control_code_from_proto(res.control_code)
        )

    def _streaming_request(self, method, batch_data, _streams_, _timeout_):
        assert _streams_, "Not implemented yet for unary requests"
        if self.request_queue is None:
            self._initialize_stream()
        ins = server_model_pb2.BatchData(
            method=method,
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )
        self.request_queue.put(ins)

    def _nonblocking_request(self, method, batch_data, _timeout_, _streams_):
        assert _streams_, "Not implemented yet for unary requests"
        if self.request_queue is None:
            self._initialize_stream()

        ins = server_model_pb2.BatchData(
            method=method,
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )
        self.request_queue.put(ins)
        def get_future_fuction():
            res = next(self.response_stream)
            batch_data = BatchData(
                data=from_grpc_format(res.data),
                control_code=control_code_from_proto(res.control_code)
            )
            return self._parse_response_args(batch_data)

        future_response = FutureResponse(get_future_fuction)
        return future_response

    def close_stream(self):
        if self.request_queue is None:
            return
        self.request_queue.put(server_model_pb2.BatchData(
            control_code=control_code_to_proto(ControlCode.DO_CLOSE_STREAM)
        ))
        res = next(self.response_stream)
        assert control_code_from_proto(res.control_code) == ControlCode.STREAM_CLOSED_OK


    def get_pending_batches_count(self) -> int:
        raise Exception("TODO")
