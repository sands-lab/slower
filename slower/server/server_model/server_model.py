from abc import ABC, abstractmethod

from flwr.common import GetParametersRes

from slower.common import (
    ServerModelFitIns,
    ServerModelEvaluateIns,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    BatchPredictionIns,
    BatchPredictionRes,
    DataBatchForward,
    DataBatchBackward,
    UpdateServerModelRes,
    ControlCode
)


class ServerModel(ABC):

    @abstractmethod
    def serve_prediction_request(
        self,
        batch: BatchPredictionIns
    ) -> BatchPredictionRes:
        """Compute the prediction for the given embeddings using the server-side model

        Parameters
        ----------
        batch : BatchPredictionIns
            A batch of data containing the embeddings as computed by
            the client model.

        Returns
        -------
        BatchPredictionRes
            Final predictions as computed by the server model.
        """

    @abstractmethod
    def serve_gradient_update_request(
        self,
        batch: GradientDescentDataBatchIns
    ) -> GradientDescentDataBatchRes:
        """Update the server model and return the gradient information
        used by the client to finish backpropagating the error

        Parameters
        ----------
        batch : GradientDescentDataBatchIns
            A batch of data containing the embeddings as computed by
            the client model and the target labels.

        Returns
        -------
        GradientDescentDataBatchRes
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
    def configure_evaluate(self, ins: ServerModelEvaluateIns) -> None:
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

    def u_forward(self, batch: DataBatchForward) -> DataBatchForward:
        """Forward pass in the U-shaped architecture

        Parameters
        ----------
        batch : DataBatchForward
            A batch of data containing the embeddings as computed by the client model

        Returns
        -------
        DataBatchForward
            Embeddings as computed with the server model to be sent to the client
        """

    def u_backward(self, batch_gradient: DataBatchBackward) -> DataBatchBackward:
        """Backward pass in the U-shaped architecture

        Parameters
        ----------
        gradient : DataBatchBackward
            Gradient information sent by the client

        Returns
        -------
        DataBatchBackward
            Gradient information obtained after backpropagating the error through the server model
        """

    def get_synchronization_result(self) -> UpdateServerModelRes:
        """Get return data to the client when the client asks to synchronize the stream

        Returns
        -------
        UpdateServerModelRes
            _description_
        """
        return UpdateServerModelRes(ControlCode.STREAM_CLOSED_OK, b"")
