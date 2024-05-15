
from abc import ABC
from typing import Callable, Dict, Iterator

import numpy as np
from flwr.common import (
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    GetParametersRes,
    Status,
    Code,
    bytes_to_ndarray,
    ndarray_to_bytes
)

from slower.server.server_model.server_model import ServerModel
from slower.common import (
    ServerModelFitIns,
    ServerModelEvaluateIns,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    BatchPredictionIns,
    BatchPredictionRes,
    ControlCode,
    UpdateServerModelRes,
    DataBatchForward,
    DataBatchBackward,
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
        embeddings: np.ndarray
    ) -> bytes:
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
        _ = (self, embeddings)
        return np.empty(0,)

    def serve_gradient_update_request(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
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
        _ = (self, embeddings, labels)
        return np.empty(0,)

    def u_forward(self, embeddings: np.ndarray) -> np.ndarray:
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
        _ = (embeddings, )
        return np.empty(0,)

    def u_backward(self, gradient: np.ndarray) -> np.ndarray:
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
        _ = (gradient, )
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
    batch: GradientDescentDataBatchIns
) -> GradientDescentDataBatchRes:
    grad = self.numpy_server_model.serve_gradient_update_request(
        embeddings=bytes_to_ndarray(batch.embeddings),
        labels=bytes_to_ndarray(batch.labels)
    )
    grad = ndarray_to_bytes(grad)
    return GradientDescentDataBatchRes(gradient=grad, control_code=ControlCode.OK)


def _serve_prediction_request(self, batch: BatchPredictionIns):
    predictions = self.numpy_server_model.serve_prediction_request(
        embeddings=bytes_to_ndarray(batch.embeddings)
    )
    predictions = ndarray_to_bytes(predictions)
    return BatchPredictionRes(predictions=predictions, control_code=ControlCode.OK)


def _serve_u_forward(self, batch: DataBatchForward):
    embeddings = self.numpy_server_model.u_forward(
        embeddings=bytes_to_ndarray(batch.embeddings)
    )
    embeddings = ndarray_to_bytes(embeddings)
    return DataBatchForward(embeddings=embeddings, control_code=batch.control_code)

def _serve_u_backward(self, batch_gradient: DataBatchBackward):
    gradient = self.numpy_server_model.u_backward(
        gradient=bytes_to_ndarray(batch_gradient.gradient)
    )
    gradient = ndarray_to_bytes(gradient)
    return DataBatchBackward(gradient=gradient, control_code=batch_gradient.control_code)


def _get_synchronization_result(self) -> UpdateServerModelRes:

    result = self.numpy_server_model.get_synchronization_result()  # type: ignore
    if not isinstance(result, np.ndarray):
        raise Exception(f"get_synchronization_result must return a ndarray, got {type(result)} instead")
    return UpdateServerModelRes(
        control_code=ControlCode.STREAM_CLOSED_OK,
        result=ndarray_to_bytes(result)
    )


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
