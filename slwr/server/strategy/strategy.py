from abc import abstractmethod
from typing import Optional, List, Tuple, Dict, Callable

from flwr.common import Parameters, Scalar
from flwr.server.strategy import Strategy as FlwrStrategy

from slwr.server.server_model.server_model import ServerModel
from slwr.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)
from slwr.server.server_model.utils import ClientRequestGroup


class Strategy(FlwrStrategy):

    @abstractmethod
    def init_server_model_fn(self) -> ServerModel:
        """Create a new ServerModel

        Parameters
        ----------

        ServerModel
        -------
        bool
            A new instance of a ServerModel
        """

    def initialize_server_parameters(
        self
    ) -> Optional[Parameters]:
        """Initialize the (global) client-side model parameters

        Returns
        -------
        Optional[Parameters]
            If parameters are return, the server will threat these as the
            initial server-side global model parameters
        """
        return None

    def initialize_parameters(self, client_manager):
        return None

    @abstractmethod
    def configure_server_fit(
        self,
        server_round: int,
        parameters: Parameters,
        cids: List[str],
    ) -> List[ServerModelFitIns]:
        """Configure the server model for the next round of training

        Parameters
        ----------
        server_round : int
            The current round of split learning (epoch)
        parameters : Parameters
            The current version of the server-side model parameters
        cids : List[str]
            List of client IDs selected for this round of training

        Returns
        -------
        ServerModelFitIns
            The instructions for the next round of training
        """

    @abstractmethod
    def configure_server_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        cids: List[str],
    ) -> List[ServerModelEvaluateIns]:
        """Configure the server model for the next round of evaluation

        Parameters
        ----------
        server_round : int
            The current round of split learning
        parameters : Parameters
            The current version of the server-side model parameters
        cids : List[str]
            List of client IDs selected for this round of evaluation

        Returns
        -------
        ServerModelEvaluateIns
            The configuration for the next round of model evaluation
        """

    @abstractmethod
    def aggregate_server_fit(
        self,
        server_round: int,
        results: List[ServerModelFitRes]
    ) -> Optional[Parameters]:
        """Aggregate the server-side training results

        Parameters
        ----------
        server_round : int
            The current round of split learning
        results : List[ServerModelFitRes]
            Successful updates from previously active server model

        Returns
        -------
        Optional[Parameters]
            New version of the server-side model parameters. If None is returned, the
            server-side model parameters are not updated
        """

    # pylint: disable=arguments-differ
    def evaluate(
        self,
        server_round: int,
        client_parameters: Parameters,
        server_parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current client+server parameters on the server

        Parameters
        ----------
        server_round : int
            server_round
        client_parameters : Parameters
            parameters of the server model
        server_parameters : Parameters
            parameters of the server model

        Returns
        -------
        Optional[Tuple[float, Dict[str, Scalar]]]
            metrics
        """

    @abstractmethod
    def route_client_request(
        self,
        cid: str,
        method_name: str,
    ) -> Tuple[ClientRequestGroup, Optional[Callable]]:
        """Map a client ID to a RequestGroup that will contain the request

        Parameters
        ----------
        cid : str
            ID of the client
        method_name : str
            Name of the method invoked by the client

        Returns
        -------
        Tuple[ClientRequestGroup, Callable]
            Tuple containing a the ClientRequestGroup, that will contain the request, and a
            callback, which is executed after the ClientRequestGroup has been processed. Note,
            that the strategy needs to handle creating new ClientRequestGroups.
        """

    @abstractmethod
    def mark_ready_requests(self) -> None:
        """Mark the requests as ready to be served
        """

    @abstractmethod
    def mark_client_as_done(self, cid: str) -> None:
        """Mark the client as done
        """
