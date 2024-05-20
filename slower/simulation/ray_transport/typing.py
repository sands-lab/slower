from typing import Union

from flwr.common import (
    GetParametersRes,
)

from slower.common import BatchData



# All possible returns by the server models
ServerRes = Union[
    BatchData, GetParametersRes
]
