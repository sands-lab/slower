from abc import abstractmethod
from typing import Optional, List

from flwr.common import Parameters
from flwr.server.strategy import Strategy

from slower.server.server_model.server_model import ServerModel
from slower.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)


class SlStrategy(Strategy):

    @abstractmethod
    def has_common_server_model(self) -> bool:
        """Return true if all the clients server model segment is common to all the clients that
        participate in the training round. If true, a single server-side model segment will be
        created and will server all clients' requests. The order with which the data will be
        processed depends on the order with which the requests are received by the server.
        If set to false, each client will train its own model server-side model segment. These
        trained model segments will be aggregated at the end of the current server round.

        Parameters
        ----------

        Returns
        -------
        bool
            True if the server-side model segment is common to all the clients in the round, false
            otherwise
        """

    @abstractmethod
    def init_server_model_fn(self) -> ServerModel:
        """Create a new ServerModel.

        Parameters
        ----------

        ServerModel
        -------
        bool
            A new instance of a ServerModel
        """

    @abstractmethod
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

    @abstractmethod
    def configure_server_fit(
        self,
        server_round: int,
        parameters: Parameters
    ) -> ServerModelFitIns:
        """Configure the server-side model segment for the next round of training

        Parameters
        ----------
        server_round : int
            The current round of split learning
        parameters : Parameters
            The current version of the server-side model parameters

        Returns
        -------
        ServerModelFitIns
            The instructions for the next round of training. TODO: as of now, all the clients
            interact with the same initial version of the server-model segment, though it might
            make sense to consider giving more flexibility
        """

    @abstractmethod
    def configure_server_evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> ServerModelEvaluateIns:
        """Configure the server-side model segment for the next round of evaluation

        Parameters
        ----------
        server_round : int
            The current round of split learning
        parameters : Parameters
            The current version of the server-side model parameters

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
            Successful updates from previously active server model segments

        Returns
        -------
        Optional[Parameters]
            New version of the server-side model parameters. If None is returned, the
            server-side model parameters are not updated
        """
