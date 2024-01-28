from typing import Union, Callable

from flwr.common import (
    GetParametersRes,
    FitRes,
    EvaluateRes
)

from slower.client.client import Client
from slower.common import (
    BatchPredictionRes,
    GradientDescentDataBatchRes
)


# All possible returns by a client
ClientRes = Union[
    GetParametersRes, FitRes, EvaluateRes
]
# A function to be executed by a client to obtain some results
JobFn = Callable[[Client], ClientRes]

# All possible returns by the server model segment
ServerRes = Union[
    BatchPredictionRes, GradientDescentDataBatchRes, GetParametersRes
]
