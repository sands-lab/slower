from queue import SimpleQueue
from typing import Optional

from grpc import Future
from flwr.common import GetParametersRes

from slower.common import ServerModelEvaluateIns, ServerModelFitIns, ControlCode, BatchData
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
        self.u_future: Optional[Future[BatchData]] = None

    def serve_prediction_request(
        self,
        batch_data: BatchData,
        timeout: Optional[float]
    ) -> BatchData:
        ins = server_model_pb2.BatchData(
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )
        res = self.stub.ServePredictionRequest(ins)
        return BatchData(
            data=from_grpc_format(res.data),
            control_code=control_code_from_proto(res.control_code)
        )

    def serve_gradient_update_request(
        self,
        batch_data: BatchData,
        timeout: Optional[float]
    ) -> BatchData:
        ins = server_model_pb2.BatchData(
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )

        res = self.stub.ServeGradientUpdateRequest(ins)
        return BatchData(
            data=from_grpc_format(res.data),
            control_code=control_code_from_proto(res.control_code)
        )

    def update_server_model(
        self,
        batch_data: BatchData,
    ) -> None:
        if self.request_queue is None:
            self.request_queue = SimpleQueue()
            queue_iterator = iter(self.request_queue.get, None)
            self.request_future = self.stub.UpdateServerModelRequests.future(queue_iterator)
        ins = server_model_pb2.BatchData(
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )
        self.request_queue.put(ins)

    def close_stream(self) -> BatchData:
        if self.request_queue is None:
            raise ClientException("Trying to close_stream but stream is not open!")
        self.request_queue.put(server_model_pb2.BatchData(
            data={},
            control_code=server_model_pb2.ControlCode.DO_CLOSE_STREAM
        ))
        result = self.request_future.result()
        self.request_queue, self.request_future = None, None
        return BatchData(
            data=from_grpc_format(result.data),
            control_code=control_code_from_proto(result.control_code),
        )

    def get_parameters(
        self,
        timeout: Optional[float]
    ) -> GetParametersRes:
        raise ClientException("Client tried to get parameters from the server")

    def configure_fit(
        self,
        ins: ServerModelFitIns,
        timeout: Optional[float]
    ) -> None:
        raise ClientException("Client tried to configure server fit")


    def configure_evaluate(
        self,
        ins: ServerModelEvaluateIns,
        timeout: Optional[float]
    ) -> None:
        raise ClientException("Client tried to configure server evaluate")

    def u_forward(
        self,
        batch_data: BatchData
    ) -> BatchData:
        if self.u_future is not None:
            # wait to get the response from the server, signaling that the server has
            # completed the backward pass and hence the client can send the next request
            cc = control_code_from_proto(self.u_future.result().control_code)
            if cc != ControlCode.OK:
                print("Warning: backward pass concluded with control message", cc)
        ins = server_model_pb2.BatchData(
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )
        res = self.stub.UForward(ins)
        return BatchData(
            data=from_grpc_format(res.data),
            control_code=control_code_from_proto(res.control_code)
        )

    def u_backward(
        self,
        batch_data: BatchData,
        blocking=True
    ) -> BatchData:
        ins = server_model_pb2.BatchData(
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code)
        )
        res = None
        if blocking:
            res = self.stub.UBackward(ins)
            res = BatchData(
                data=from_grpc_format(res.data),
                control_code=control_code_from_proto(res.control_code)
            )
        else:
            self.u_future = self.stub.UBackward.future(ins)
        return res

    def _get_synchronization_result(self):
        raise ClientException("Should not be called by the client!!")

    def get_pending_batches_count(self) -> int:
        if self.request_queue is None:
            raise ClientException("Trying to close_stream but stream is not open!")
        return self.request_queue.qsize()
