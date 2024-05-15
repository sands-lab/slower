from dataclasses import dataclass
from enum import Enum
from typing import Dict

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
class GradientDescentDataBatchIns:
    """
    Instructions for a single GD update computation of the server model
    """
    embeddings: bytes
    labels: bytes
    control_code: ControlCode


@dataclass
class GradientDescentDataBatchRes:
    """
    Gradient data returned by the server model. This data should be used for backpropagation on
    the error on client side until the first layer
    """
    gradient: bytes
    control_code: ControlCode


@dataclass
class BatchPredictionIns:
    """
    Batch of data for which the client wants to get the final predictions (that is, the server
    does not train using the data)
    """
    embeddings: bytes
    control_code: ControlCode


@dataclass
class BatchPredictionRes:
    """
    Final client predictions for a batch of data
    """
    predictions: bytes
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


@dataclass
class UpdateServerModelRes:
    control_code: ControlCode
    result: bytes


@dataclass
class DataBatchForward:
    """Object used during the forward propagation in the U-shaped architecture
    """
    embeddings: bytes
    control_code: ControlCode


@dataclass
class DataBatchBackward:
    """Object used during the backward propagation in the U-shaped architecture
    """
    gradient: bytes
    control_code: ControlCode
