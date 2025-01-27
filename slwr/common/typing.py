from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union, List, Optional

from flwr.common import Config, NDArrays


class ControlCode(Enum):
    OK = 0
    DO_CLOSE_STREAM = 1
    STREAM_CLOSED_OK = 2
    ERROR_PROCESSING_STREAM = 3
    INIT_STREAM = 4


class RequestType(Enum):
    """
    Possible types of the request a client can make to the server:
    - BLOCKING: the client issues the request and waits until receiving the return value
    - FUTURE: the client issues a request and continues processing. The client must fetch the result from the server before submitting a new request
    - STRAM: the client issues requests and does not wait to receive any response
    """
    BLOCKING = 0
    FUTURE = 1
    STREAM = 2


class RequestArgumentFormat(Enum):
    """
    Server methods can be requested by passing either the native BatchData object or
    (preferred way) using either bytes or numpy/pytorch tensors
    """
    RAW = 1  # BatchData
    NUMPY = 2  # Numpy arrays / list of numpy arrays
    TORCH = 3  # Torch arrays / list of Torch arrays


@dataclass
class BatchData:
    """
    Generic class being passed between client and server during training/evaluation
    """
    data: Dict[str, Union[bytes, List[bytes]]]
    control_code: ControlCode


@dataclass
class ServerModelFitIns:
    """
    Configuration for the server model prior training:
    - `parameters` contains the currect weights of the server model
    - `config` may contain arbitrary user-defined values (learning rate, optimizer, ...)
    """
    parameters: NDArrays
    config: Config
    sid: str


@dataclass
class ServerModelEvaluateIns:
    """
    Configuration for the server model of the model prior evaluation
    """
    parameters: NDArrays
    config: Config
    sid: str


@dataclass
class ServerModelFitRes:
    """
    Summary of the final version of the server model, with the updated model weights (`parameters`)
    and the number of training example of the server model has processed
    """
    parameters: NDArrays
    config: Config
    sid: Optional[str] = None

    def __post_init__(self):
        # user should not set the sid, as it is automatically set by the framework
        assert self.sid is None
