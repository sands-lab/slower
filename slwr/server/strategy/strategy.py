from abc import abstractmethod
from typing import Optional, List, Tuple, Dict

from flwr.common import Parameters, Scalar
from flwr.server.strategy import Strategy as FlwrStrategy

from slwr.server.server_model.server_model import ServerModel
from slwr.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)


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
    def cid_to_server_model(
        self,
        cid: str,
        request_queues: Dict[str, List[str]],
        method: str,
    ) -> Tuple[str, bool]:
        """Map a client ID to a server model to be used to serve the given request

        Parameters
        ----------
        cid : str
            ID of the client
        request_queues : Dict[str, List[str]]
            Mapping of server ID to list of client IDs that the server model with the
            given SID is waiting to serve

        Returns
        -------
        Tuple[bool, str]
            Tuple containing a the ID (sid) of the server model to be used to serve the given
            request. The second value in the tuple is a boolean indicating whether the server
            model is allowed to start executing the batch of requests. For instance, if cid is
            "4", the selected sid is "0" and request_queues["0"] equals ["1","2","3"], if
            returning ("0", True), the server model with sid "0" will start executing the batch
            of requests sent by clients ["1", "2", "3", "4"]. If returning ("0", False), the
            server model with sid "0" will wait until the next request is received, and the
            process will repeat.
        """
