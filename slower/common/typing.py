from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union, List

from flwr.common import Parameters, Scalar


@dataclass
class ControlCode(Enum):
    OK = 0
    DO_CLOSE_STREAM = 1
    STREAM_CLOSED_OK = 2
    ERROR_PROCESSING_STREAM = 3

    def __eq__(self, other):
        return self.value == other.value


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
    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class ServerModelEvaluateIns:
    """
    Configuration for the server model of the model prior evaluation
    """
    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class ServerModelFitRes:
    """
    Summary of the final version of the server model, with the updated
    model weights (`parameters`), the number of training example of the corresponding client
    (`num_exampple`), and the `cid` of the corresponding client
    """
    parameters: Parameters
    num_examples: int
    cid: str
