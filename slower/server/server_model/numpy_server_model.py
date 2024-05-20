
from abc import ABC
from typing import Callable, Dict, Union, List

import numpy as np
from flwr.common import (
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    GetParametersRes,
    Status,
    Code,
)

from slower.server.server_model.server_model import ServerModel
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

    def serve_prediction_request(
        self,
        batch_data: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Compute the prediction for the given embeddings using the server model

        Parameters
        ----------
        embeddings : bytes
            The embeddings as computed by the client model for some batch of data.

        Returns
        -------
        nd.ndarray
            Final predictions as computed by the server model.
        """
        _ = (self, batch_data)
        return np.empty(0,)

    def serve_gradient_update_request(
        self,
        batch_data: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Update the server model and return the gradient information
        used by the client to finish backpropagating the error

        Parameters
        ----------
        embeddings : bytes
            A batch of data containing the embeddings as computed by
            the client model for the current batch.
        labels : bytes
            The target labels of the current batch.

        Returns
        -------
        bytes
            Gradient information used by the client for finishing the backpropagation.
        """
        _ = (self, batch_data)
        return {}

    def u_forward(
        self,
        batch_data: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Perform the forward pass on the server

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings as computed by the first part of the client-side model

        Returns
        -------
        np.ndarray
            Embeddings as computed by the server-side model
        """
        _ = (batch_data,)
        return {}

    def u_backward(
        self,
        batch_data: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Run the back propagation on the server model

        Parameters
        ----------
        gradient : np.ndarray
            Gradient information as computed by the final client layers

        Returns
        -------
        np.ndarray
            Gradient computed after backpropagating until the first server layer
        """
        _ = (batch_data, )
        return np.empty(0,)

    def to_server_model(self) -> ServerModel:
        """Convert to object to Client type and return it."""
        return _wrap_numpy_server_model(server_model=self)

    def get_synchronization_result(self) -> np.ndarray:
        return np.empty((0,))


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


def _serve_gradient_update_request(
    self: ServerModel,
    batch_data: BatchData
) -> BatchData:
    res = self.numpy_server_model.serve_gradient_update_request(
        batch_data=bytes_to_ndarray_dict(batch_data.data)
    )
    res = ndarray_dict_to_bytes(res)
    return BatchData(data=res, control_code=ControlCode.OK)


def _serve_prediction_request(
    self,
    batch_data: BatchData
) -> BatchData:
    res = self.numpy_server_model.serve_prediction_request(
        batch_data=bytes_to_ndarray_dict(batch_data.data)
    )
    res = ndarray_dict_to_bytes(res)
    return BatchData(data=res, control_code=ControlCode.OK)


def _serve_u_forward(
    self,
    batch_data: BatchData
) -> BatchData:
    res = self.numpy_server_model.u_forward(
        batch_data=bytes_to_ndarray_dict(batch_data.data)
    )
    res = ndarray_dict_to_bytes(res)
    return BatchData(data=res, control_code=ControlCode.OK)

def _serve_u_backward(self, batch_data: BatchData):
    res = self.numpy_server_model.u_backward(
        batch_data=bytes_to_ndarray_dict(batch_data.data)
    )
    res = ndarray_dict_to_bytes(res)
    return BatchData(data=res, control_code=ControlCode.OK)


def _get_synchronization_result(self) -> BatchData:

    res = self.numpy_server_model.get_synchronization_result()
    res = ndarray_dict_to_bytes(res)
    return BatchData(data=res, control_code=ControlCode.STREAM_CLOSED_OK)


def _wrap_numpy_server_model(
    server_model: NumPyServerModel
) -> ServerModel:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
        "serve_prediction_request": _serve_prediction_request,
        "serve_gradient_update_request": _serve_gradient_update_request,
        "configure_fit": _configure_fit,
        "configure_evaluate": _configure_evaluate,
        "get_parameters": _get_parameters,
        "u_forward": _serve_u_forward,
        "u_backward": _serve_u_backward,
        "get_synchronization_result": _get_synchronization_result,
    }

    # pylint: disable=abstract-class-instantiated
    wrapper_class = type("NumPyServerModelWrapper", (ServerModel,), member_dict)


    # Create and return an instance of the newly created class
    return wrapper_class(numpy_server_model=server_model)  # type: ignore
