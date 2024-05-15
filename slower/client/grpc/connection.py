import uuid
import sys
from logging import DEBUG
from queue import Queue
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator, Optional, Union, Callable, Tuple

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.common.grpc import create_channel
from flwr.common.address import parse_address
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.proto.transport_pb2_grpc import FlowerServiceStub
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import TaskIns, TaskRes, Task
from flwr.client.grpc_client.connection import on_channel_state_change

from slower.proto.server_model_pb2_grpc import ServerModelStub


def init_connection(server_address: str):
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"
    return grpc_connection, address


@contextmanager
def grpc_connection(
    server_address: str,
    insecure: bool,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
) -> Iterator[
    Tuple[
        Callable[[], Optional[TaskIns]],
        Callable[[TaskRes], None],
        Optional[Callable[[], None]],
        Optional[Callable[[], None]],
    ]
]:
    if isinstance(root_certificates, str):
        root_certificates = Path(root_certificates).read_bytes()

    channel = create_channel(
        server_address=server_address,
        insecure=insecure,
        root_certificates=root_certificates,
        max_message_length=max_message_length,
    )

    channel.subscribe(on_channel_state_change)

    queue: Queue[ClientMessage] = Queue(  # pylint: disable=unsubscriptable-object
        maxsize=1
    )
    stub = FlowerServiceStub(channel)
    server_model_servicer_stub = ServerModelStub(channel)

    server_message_iterator: Iterator[ServerMessage] = stub.Join(iter(queue.get, None))

    def receive() -> TaskIns:
        server_message = next(server_message_iterator)
        return TaskIns(
            task_id=str(uuid.uuid4()),
            group_id="",
            workload_id=0,
            task=Task(
                producer=Node(node_id=0, anonymous=True),
                consumer=Node(node_id=0, anonymous=True),
                ancestry=[],
                legacy_server_message=server_message,
            ),
        )

    def send(task_res: TaskRes) -> None:
        msg = task_res.task.legacy_client_message
        return queue.put(msg, block=False)

    try:
        # Yield methods
        yield (receive, send, server_model_servicer_stub)
    finally:
        # Make sure to have a final
        channel.close()
        log(DEBUG, "gRPC channel closed")
