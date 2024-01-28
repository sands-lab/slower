from io import BytesIO
from typing import cast, List
from logging import INFO

from flwr.common.logger import log


try:
    import torch
except ImportError:
    log(INFO, "`torch` library could not be imported")


def bytes_to_torch(data: bytes, requires_grad: bool) -> torch.Tensor:
    buffer = BytesIO(data)
    tensor = torch.load(buffer)
    tensor.requires_grad_(requires_grad)
    return cast(torch.Tensor, tensor)


def torch_to_bytes(data: torch.Tensor) -> bytes:
    buffer = BytesIO()
    torch.save(data.detach().cpu(), buffer)
    return buffer.getvalue()


def torch_list_to_bytes(data: List[torch.Tensor]) -> bytes:
    data = [d.detach().cpu() for d in data]
    buffer = BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def bytes_to_torch_list(data: bytes):
    buffer = BytesIO(data)
    tensor = torch.load(buffer)
    return cast(List[torch.Tensor], tensor)
