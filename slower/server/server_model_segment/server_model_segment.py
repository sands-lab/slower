from abc import ABC, abstractmethod

from flwr.common import GetParametersRes

from slower.common import (
    ServerModelSegmentFitIns,
    ServerModelSegmentEvaluateIns,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    BatchPredictionIns,
    BatchPredictionRes
)



class ServerModelSegment(ABC):

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
            the client-side segment of the model.

        Returns
        -------
        BatchPredictionRes
            Final predictions as computed by the server-side segment of the model.
        """


    @abstractmethod
    def serve_gradient_update_request(
        self,
        batch: GradientDescentDataBatchIns
    ) -> GradientDescentDataBatchRes:
        """Update the server-side segment of the model and return the gradient information
        used by the client to finish backpropagating the error

        Parameters
        ----------
        batch : GradientDescentDataBatchIns
            A batch of data containing the embeddings as computed by
            the client-side segment of the model and the target labels.

        Returns
        -------
        GradientDescentDataBatchRes
            Gradient information used by the client for finishing the backpropagation.
        """

    @abstractmethod
    def get_parameters(self) -> GetParametersRes:
        """Return the current parameters of the server-side segment of the model

        Returns
        -------
        GetParametersRes
            The current local model parameters.
        """


    @abstractmethod
    def configure_fit(
        self,
        ins: ServerModelSegmentFitIns
    ) -> None:
        """Configure the server-side segment of the model before any client starts to train it

        Parameters
        ----------
        ins : ServerModelSegmentFitIns
            The training instructions containing the current version of the
            server-side segment of the model, and a dictionary of configuration
            values used to customize the training process (learning rate, optimizer, ...).

        Returns
        -------
        None
        """

    @abstractmethod
    def configure_evaluate(self, ins: ServerModelSegmentEvaluateIns) -> None:
        """Configure the server-side segment of the model before any client starts to make
        predictions using it

        Parameters
        ----------
        ins : ServerModelSegmentEvaluateIns
            The evaluation instructions containing the current version of the
            server-side segment of the model, and a dictionary of configuration
            values used to customize the evaluation process.

        Returns
        -------
        None
        """

    def to_server_model_segment(self):
        return self
