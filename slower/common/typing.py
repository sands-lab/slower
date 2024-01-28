from dataclasses import dataclass
from typing import Dict

from flwr.common import Parameters, Scalar


@dataclass
class GradientDescentDataBatchIns:
    """
    Instructions for a single GD update computation of the server model segment
    """
    embeddings: bytes
    labels: bytes


@dataclass
class GradientDescentDataBatchRes:
    """
    Gradient data returned by the server model. This data should be used for backpropagation on
    the error on client side until the first layer
    """
    gradient: bytes


@dataclass
class BatchPredictionIns:
    """
    Batch of data for which the client wants to get the final predictions (that is, the server
    does not train using the data)
    """
    embeddings: bytes


@dataclass
class BatchPredictionRes:
    """
    Final client predictions for a batch of data
    """
    predictions: bytes


@dataclass
class ServerModelSegmentFitIns:
    """
    Configuration for the server side segment of the model prior training:
    - `parameters` contains the currect weights of the server-side segment of the model
    - `config` may contain arbitrary user-defined values (learning rate, optimizer, ...)
    """
    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class ServerModelSegmentEvaluateIns:
    """
    Configuration for the server side segment of the model prior evaluation
    """
    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class ServerModelSegmentFitRes:
    """
    Summary of the final version of the server side segment of the model, with the updated
    model weights (`parameters`), the number of training example of the corresponding client
    (`num_exampple`), and the `cid` of the corresponding client
    """
    parameters: Parameters
    num_examples: int
    cid: str
