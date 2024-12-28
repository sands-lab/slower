from typing import AsyncIterable
import asyncio

from slwr.proto import server_model_pb2_grpc
from slwr.proto import server_model_pb2
from slwr.server.server_model.manager import ServerModelManager
from slwr.common import BatchData, ControlCode
from slwr.common.serde import (
    control_code_from_proto,
    control_code_to_proto,
    from_grpc_format,
    to_grpc_format
)


class EventWithReturnValue(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.result = None

    def set_result(self, result: BatchData):
        self.result = result
        self.set()

    def get_result(self) -> BatchData:
        return self.result


# pylint: disable=no-member
class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):
    def __init__(self, server_model_manager: ServerModelManager) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager
        self._lock = asyncio.Lock()
        self.request_queues = {}

    async def _compute_response(
        self,
        cid: str,
        request: server_model_pb2.BatchData
    ) -> BatchData:
        data = BatchData(
            data = from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code)
        )
        method_name = request.method
        event = EventWithReturnValue()
        async with self._lock:
            current_request_status = {k: len(v) for k, v in self.request_queues.items()}
            sid, trigger = self.server_model_manager.strategy.cid_to_server_model(
                cid,
                current_request_status,
                method_name,
            )
            if (sid, method_name) not in self.request_queues:
                self.request_queues[(sid, method_name)] = []
            self.request_queues[(sid, method_name)].append((data, event))
            if trigger:
                baches, events = zip(*self.request_queues[(sid, method_name)])
                self.request_queues[(sid, method_name)] = []

        if trigger:
            self._trigger_computation(sid, method_name, list(baches), list(events))

        await event.wait()
        return event.get_result()

    def _trigger_computation(self, sid, method_name, baches, events):
        server_model = self.server_model_manager.get_server_model(sid)
        method = getattr(server_model, method_name)
        res = method(baches)

        for e, r in zip(events, res):
            e.set_result(r)

    def _to_grpc(self, data: BatchData) -> server_model_pb2.BatchData:
        return server_model_pb2.BatchData(
            data=to_grpc_format(data.data),
            control_code=control_code_to_proto(data.control_code)
        )

    async def UnaryRequest(
        self,
        request,
        context
    ) -> server_model_pb2.BatchData:
        _ = (context, )
        return self._to_grpc(await self._compute_response(request.cid, request))

    async def StreamRequest(
        self,
        request_iterator: AsyncIterable[server_model_pb2.BatchData],
        context,
    ) -> AsyncIterable[server_model_pb2.BatchData]:
        _ = (context, )
        req = await request_iterator.__anext__()
        assert control_code_from_proto(req.control_code) == ControlCode.INIT_STREAM
        cid = req.cid
        async for req in request_iterator:
            if control_code_from_proto(req.control_code) == ControlCode.DO_CLOSE_STREAM:
                break
            res = await self._compute_response(cid, req)

            if res is not None:
                yield self._to_grpc(res)
        yield server_model_pb2.BatchData(
            control_code=control_code_to_proto(ControlCode.STREAM_CLOSED_OK)
        )
