from abc import ABC, abstractmethod

from flwr.common import NDArrays

from slwr.common import (
    ServerModelFitIns,
    ServerModelEvaluateIns,
    ServerModelFitRes,
)


class ServerModel(ABC):

    @abstractmethod
    def get_parameters(self) -> NDArrays:
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
    def get_fit_result(self) -> ServerModelFitRes:
        """Method called after the current server model has finished training

        Returns
        -------
        ServerModelFitRes
            Parameters and possible cofiguration to be used for aggregating the server model weights
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
