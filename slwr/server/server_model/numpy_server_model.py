from abc import ABC
from functools import partial
from typing import Callable, Dict, Optional, Tuple, List, Union
import inspect

from flwr.common import (
    NDArrays,
    NDArray,
    GetParametersRes,
    Config,
)

from slwr.server.server_model.server_model import ServerModel
from slwr.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict
from slwr.common import (
    ServerModelFitIns,
    ServerModelEvaluateIns,
    ControlCode,
    BatchData,
    ServerModelFitRes,
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

    def get_fit_result(self) -> Tuple[NDArrays, Config]:
        """Return the weights of the model after training possibly with additional metadata
        to be used for aggregating the server models (e.g., the number of data points the
        server model has processed)

        Returns
        -------
        Tuple[NDArrays, Config]
            Updated weigts with configuration
        """
        return [], {}

    def configure_fit(
        self,
        parameters: NDArrays,
        config: Config,
    ) -> None:
        """Configure the server model before any client starts to train it

        Parameters
        ----------
        parameters : NDArrays
            The current weights of the global server model
        config : Config
            Additional configuration for training (learning rate, optimizer, ...)

        Returns
        -------
        None
        """
        _ = (self, parameters, config)

    def configure_evaluate(
        self,
        parameters: NDArrays,
        config: Config
    ) -> None:
        """Configure the server model before any client starts to make
        predictions using it

        Parameters
        ----------
        parameters : NDArrays
            The current weights of the global server model
        config : Config
            Additional configuration for evaluation

        Returns
        -------
        None
        """
        _ = (self, parameters, config)

    def to_server_model(self) -> ServerModel:
        """Convert to object to Client type and return it."""
        return _wrap_numpy_server_model(server_model=self)


def _constructor(
    self: ServerModel,
    numpy_server_model: NumPyServerModel
) -> None:
    self.numpy_server_model = numpy_server_model  # type: ignore


def _get_parameters(self: ServerModel) -> NDArrays:
    """Return the current local model parameters."""
    parameters = self.numpy_server_model.get_parameters()  # type: ignore
    return parameters


def _configure_fit(self: ServerModel, ins: ServerModelFitIns) -> GetParametersRes:
    self.numpy_server_model.configure_fit(
        parameters=ins.parameters,
        config=ins.config
    )


def _get_fit_result(self: ServerModel) -> ServerModelFitRes:
    params, config = self.numpy_server_model.get_fit_result()
    return ServerModelFitRes(
        parameters=params,
        config=config
    )


def _configure_evaluate(
    self: ServerModel,
    ins: ServerModelEvaluateIns
) -> GetParametersRes:
    self.numpy_server_model.configure_evaluate(
        parameters=ins.parameters,
        config=ins.config
    )


def _wrap_custom_logic(batch_data: List[BatchData], method: Callable) -> Optional[BatchData]:
    kwargs_list = [bytes_to_ndarray_dict(b.data) for b in batch_data]
    kwargs = {key: [d[key] for d in kwargs_list] for key in kwargs_list[0]}
    result: Optional[List[Dict[str, Union[NDArray, NDArrays]]]] = method(**kwargs)
    if result is None:
        return [None] * len(batch_data)

    if len(result) == 1 and len(batch_data) > 1:
        result = BatchData(
            data=ndarray_dict_to_bytes(result[0]),
            control_code=ControlCode.OK
        )
        return [result] * len(batch_data)

    assert len(result) == len(batch_data)
    return [BatchData(
        data=ndarray_dict_to_bytes(res),
        control_code=ControlCode.OK
    ) for res in result]


def _wrap_numpy_server_model(
    server_model: NumPyServerModel
) -> ServerModel:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
        "configure_fit": _configure_fit,
        "get_fit_result": _get_fit_result,
        "configure_evaluate": _configure_evaluate,
        "get_parameters": _get_parameters,
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

    # pylint: disable=abstract-class-instantiated
    wrapper_class = type("NumPyServerModelWrapper", (ServerModel,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_server_model=server_model)  # type: ignore
