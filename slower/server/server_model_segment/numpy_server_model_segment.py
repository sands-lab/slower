
from abc import ABC
from typing import Callable, Dict

from flwr.common import (
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import (
    Code,
    GetParametersRes,
    Status,
)

from slower.server.server_model_segment.server_model_segment import ServerModelSegment
from slower.common import (
    ServerModelSegmentFitIns,
    ServerModelSegmentEvaluateIns,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    BatchPredictionIns,
    BatchPredictionRes
)


class NumPyServerModelSegment(ABC):
    """Abstract base class for Flower clients using NumPy."""

    def get_parameters(self) -> NDArrays:
        """Returns the current weights of the server-side model segment

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
        """Configure the server-side segment of the model before any client starts to train it

        Parameters
        ----------
        parameters : NDArrays
            The current weights of the global server-side model segment
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
        """Configure the server-side segment of the model before any client starts to make
        predictions using it

        Parameters
        ----------
        parameters : NDArrays
            The current weights of the global server-side model segment
        config : Dict[str, Scalar]
            Additional configuration for evaluation

        Returns
        -------
        None
        """
        _ = (self, parameters, config)

    def serve_prediction_request(
        self,
        embeddings: bytes
    ) -> bytes:
        """Compute the prediction for the given embeddings using the server-side model

        Parameters
        ----------
        embeddings : bytes
            The embeddings as computed by the client-side segment of the model for
            some batch of data.

        Returns
        -------
        nd.ndarray
            Final predictions as computed by the server-side segment of the model.
        """
        _ = (self, embeddings)
        return b""

    def serve_gradient_update_request(
        self,
        embeddings: bytes,
        labels: bytes
    ) -> bytes:
        """Update the server-side segment of the model and return the gradient information
        used by the client to finish backpropagating the error

        Parameters
        ----------
        embeddings : bytes
            A batch of data containing the embeddings as computed by
            the client-side segment of the model for the current batch.
        labels : bytes
            The target labels of the current batch.

        Returns
        -------
        bytes
            Gradient information used by the client for finishing the backpropagation.
        """
        _ = (self, embeddings, labels)
        return b""

    def to_server_model_segment(self) -> ServerModelSegment:
        """Convert to object to Client type and return it."""
        return _wrap_numpy_server_model_segment(server_model_segment=self)


def _constructor(
    self: ServerModelSegment,
    numpy_server_model_segment: NumPyServerModelSegment
) -> None:
    self.numpy_server_model_segment = numpy_server_model_segment  # type: ignore


def _get_parameters(self: ServerModelSegment) -> GetParametersRes:
    """Return the current local model parameters."""
    parameters = self.numpy_server_model_segment.get_parameters()  # type: ignore
    parameters_proto = ndarrays_to_parameters(parameters)
    return GetParametersRes(
        status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
    )


def _configure_fit(self: ServerModelSegment, ins: ServerModelSegmentFitIns) -> GetParametersRes:
    self.numpy_server_model_segment.configure_fit(
        parameters=parameters_to_ndarrays(ins.parameters),
        config=ins.config
    )


def _configure_evaluate(
    self: ServerModelSegment,
    ins: ServerModelSegmentEvaluateIns
) -> GetParametersRes:
    self.numpy_server_model_segment.configure_evaluate(
        parameters=parameters_to_ndarrays(ins.parameters),
        config=ins.config
    )


def _serve_gradient_update_request(
    self: ServerModelSegment,
    batch: GradientDescentDataBatchIns
) -> GradientDescentDataBatchRes:
    grad = self.numpy_server_model_segment.serve_gradient_update_request(
        embeddings=batch.embeddings,
        labels=batch.labels
    )
    return GradientDescentDataBatchRes(gradient=grad)


def _serve_prediction_request(self, batch: BatchPredictionIns):
    predictions = self.numpy_server_model_segment.serve_prediction_request(
        embeddings=batch.embeddings
    )
    assert isinstance(predictions, bytes)
    return BatchPredictionRes(predictions=predictions)


def _wrap_numpy_server_model_segment(
    server_model_segment: NumPyServerModelSegment
) -> ServerModelSegment:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
        "serve_prediction_request": _serve_prediction_request,
        "serve_gradient_update_request": _serve_gradient_update_request,
        "configure_fit": _configure_fit,
        "configure_evaluate": _configure_evaluate,
        "get_parameters": _get_parameters,
    }

    # pylint: disable=abstract-class-instantiated
    wrapper_class = type("NumPyServerModelSegmentWrapper", (ServerModelSegment,), member_dict)


    # Create and return an instance of the newly created class
    return wrapper_class(numpy_server_model_segment=server_model_segment)  # type: ignore
