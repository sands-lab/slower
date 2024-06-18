from abc import ABC
from functools import partial
from typing import Callable, Dict, Optional
from logging import INFO
import inspect

import numpy as np
from flwr.common import (
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    GetParametersRes,
    Status,
    Code,
    log
)

from slower.server.server_model.server_model import ServerModel
from slower.common.constants import RETURN_TENSOR_TYPE_KEY
from slower.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict
from slower.common import (
    ServerModelFitIns,
    ServerModelEvaluateIns,
    ControlCode,
    BatchData,
)


class NumPyServerModel(ABC):
    """Abstract base class for Flower clients using NumPy."""

    def get_parameters(self) -> NDArrays:
        """Returns the current weights of the server model

        Returns
        -------
        NDArrays
            Numpy array with the current model weights
        """
        _ = (self,)
        return []

    def configure_fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> None:
        """Configure the server model before any client starts to train it

        Parameters
        ----------
        parameters : NDArrays
            The current weights of the global server model
        config : Dict[str, Scalar]
            Additional configuration for training (learning rate, optimizer, ...)

        Returns
        -------
        None
        """
        _ = (self, parameters, config)

    def configure_evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> None:
        """Configure the server model before any client starts to make
        predictions using it

        Parameters
        ----------
        parameters : NDArrays
            The current weights of the global server model
        config : Dict[str, Scalar]
            Additional configuration for evaluation

        Returns
        -------
        None
        """
        _ = (self, parameters, config)

    def to_server_model(self) -> ServerModel:
        """Convert to object to Client type and return it."""
        return _wrap_numpy_server_model(server_model=self)

    def get_synchronization_result(self) -> np.ndarray:
        return {}


def _constructor(
    self: ServerModel,
    numpy_server_model: NumPyServerModel
) -> None:
    self.numpy_server_model = numpy_server_model  # type: ignore


def _get_parameters(self: ServerModel) -> GetParametersRes:
    """Return the current local model parameters."""
    parameters = self.numpy_server_model.get_parameters()  # type: ignore
    parameters_proto = ndarrays_to_parameters(parameters)
    return GetParametersRes(
        status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
    )


def _configure_fit(self: ServerModel, ins: ServerModelFitIns) -> GetParametersRes:
    self.numpy_server_model.configure_fit(
        parameters=parameters_to_ndarrays(ins.parameters),
        config=ins.config
    )


def _configure_evaluate(
    self: ServerModel,
    ins: ServerModelEvaluateIns
) -> GetParametersRes:
    self.numpy_server_model.configure_evaluate(
        parameters=parameters_to_ndarrays(ins.parameters),
        config=ins.config
    )


def _get_synchronization_result(self) -> BatchData:

    res = self.numpy_server_model.get_synchronization_result()
    res = ndarray_dict_to_bytes(res)
    return BatchData(data=res, control_code=ControlCode.STREAM_CLOSED_OK)


def _wrap_custom_logic(batch_data: BatchData, method: Callable) -> Optional[BatchData]:
    kwargs = bytes_to_ndarray_dict(batch_data.data)
    res = method(**kwargs)

    if res is not None:
        if isinstance(res, dict):
            tp = "dict"
        else:
            tp = "np"
            res = {"": res}

        res[RETURN_TENSOR_TYPE_KEY] = tp
        return BatchData(
            data=ndarray_dict_to_bytes(res),
            control_code=ControlCode.OK
        )

def _wrap_numpy_server_model(
    server_model: NumPyServerModel
) -> ServerModel:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
        "configure_fit": _configure_fit,
        "configure_evaluate": _configure_evaluate,
        "get_parameters": _get_parameters,
        "get_synchronization_result": _get_synchronization_result,
    }

    parent_class = server_model.__class__.__bases__[0]
    for method_name in dir(server_model):
        if (
            method_name.startswith("_") or
            hasattr(parent_class, method_name) or
            not inspect.ismethod(getattr(server_model, method_name))
        ):
            continue
        if callable(getattr(server_model, method_name)) and method_name not in member_dict:
            method = getattr(server_model, method_name)
            member_dict[method_name] = partial(_wrap_custom_logic, method=method)
            log(INFO, f"Discovered method: {method_name}")

    # pylint: disable=abstract-class-instantiated
    wrapper_class = type("NumPyServerModelWrapper", (ServerModel,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_server_model=server_model)  # type: ignore
