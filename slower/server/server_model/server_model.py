from abc import ABC, abstractmethod

from flwr.common import GetParametersRes

from slower.common import (
    ServerModelFitIns,
    ServerModelEvaluateIns,
    BatchData,
    ControlCode,
)


class ServerModel(ABC):

    @abstractmethod
    def serve_prediction_request(
        self,
        batch_data: BatchData
    ) -> BatchData:
        """Compute the prediction for the given embeddings using the server-side model

        Parameters
        ----------
        batch : BatchData
            A batch of data containing the embeddings as computed by
            the client model.

        Returns
        -------
        BatchData
            Final predictions as computed by the server model.
        """

    @abstractmethod
    def serve_gradient_update_request(
        self,
        batch_data: BatchData
    ) -> BatchData:
        """Update the server model and return the gradient information
        used by the client to finish backpropagating the error

        Parameters
        ----------
        batch : BatchData
            A batch of data containing the embeddings as computed by
            the client model and the target labels.

        Returns
        -------
        BatchData
            Gradient information used by the client for finishing the backpropagation.
        """

    @abstractmethod
    def get_parameters(self) -> GetParametersRes:
        """Return the current parameters of the server model

        Returns
        -------
        GetParametersRes
            The current local model parameters.
        """


    @abstractmethod
    def configure_fit(
        self,
        ins: ServerModelFitIns
    ) -> None:
        """Configure the server model before any client starts to train it

        Parameters
        ----------
        ins : ServerModelFitIns
            The training instructions containing the current version of the
            server model, and a dictionary of configuration
            values used to customize the training process (learning rate, optimizer, ...).

        Returns
        -------
        None
        """

    @abstractmethod
    def configure_evaluate(
        self,
        ins: ServerModelEvaluateIns
    ) -> None:
        """Configure the server model before any client starts to make predictions using it

        Parameters
        ----------
        ins : ServerModelEvaluateIns
            The evaluation instructions containing the current version of the server model, and a
            dictionary of configuration values used to customize the evaluation process.

        Returns
        -------
        None
        """

    def to_server_model(self):
        return self

    def u_forward(
        self,
        batch_data: BatchData
    ) -> BatchData:
        """Forward pass in the U-shaped architecture

        Parameters
        ----------
        batch : BatchData
            A batch of data containing the embeddings as computed by the client model

        Returns
        -------
        BatchData
            Embeddings as computed with the server model to be sent to the client
        """

    def u_backward(
            self,
            batch_data: BatchData
        ) -> BatchData:
        """Backward pass in the U-shaped architecture

        Parameters
        ----------
        gradient : BatchData
            Gradient information sent by the client

        Returns
        -------
        BatchData
            Gradient information obtained after backpropagating the error through the server model
        """

    def get_synchronization_result(self) -> BatchData:
        """Get return data to the client when the client asks to synchronize the stream

        Returns
        -------
        BatchData
            _description_
        """
        return BatchData(data={}, control_code=ControlCode.STREAM_CLOSED_OK)
