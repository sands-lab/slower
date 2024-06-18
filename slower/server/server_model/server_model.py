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

    def get_synchronization_result(self) -> BatchData:
        """Get return data to the client when the client asks to synchronize the stream

        Returns
        -------
        BatchData
            _description_
        """
        return BatchData(data={}, control_code=ControlCode.STREAM_CLOSED_OK)
