from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Dict

from flwr.common import Parameters, FitIns, EvaluateIns, FitRes, Scalar, EvaluateRes
from flwr.server.client_manager import ClientManager

from slower.client.proxy.client_proxy import ClientProxy
from slower.server.server_model_segment.server_model_segment import ServerModelSegment
from slower.common import (
    ServerModelSegmentEvaluateIns,
    ServerModelSegmentFitIns,
    ServerModelSegmentFitRes,

)

class SlStrategy(ABC):

    @abstractmethod
    def has_common_server_model_segment(self) -> bool:
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
    def init_server_model_segment_fn(self) -> ServerModelSegment:
        """Create a new ServerModelSegment.

        Parameters
        ----------

        ServerModelSegment
        -------
        bool
            A new instance of a ServerModelSegment
        """

    @abstractmethod
    def evaluate(
        self,
        server_round: int,
        client_parameters: Parameters,
        server_parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the currently available model weights (including the server-side and the
        client-side model weights) on a centrally available data

        Parameters
        ----------
        server_round : int
            Index of the current server round
        client_parameters : Parameters
            Current version of the client-side model parameters
        server_parameters : Parameters
            Current version of the server-side model parameters

        Returns
        -------
        Optional[Tuple[float, Dict[str, Scalar]]]
            The evaluation result, usually a Tuple containing loss and a
            dictionary containing task-specific metrics (e.g., accuracy).
        """

    @abstractmethod
    def initialize_client_parameters(
        self,
        client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) client-side model parameters.

        Parameters
        ----------
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            initial client-side global model parameters.
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
    def configure_client_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Parameters
        ----------
        server_round : int
            The current round of split learning.
        parameters : Parameters
            The current (global) client-side model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        fit_configuration : List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of split learning.
        """

    @abstractmethod
    def configure_server_fit(
        self,
        server_round: int,
        parameters: Parameters
    ) -> ServerModelSegmentFitIns:
        """Configure the server-side model segment for the next round of training

        Parameters
        ----------
        server_round : int
            The current round of split learning
        parameters : Parameters
            The current version of the server-side model parameters

        Returns
        -------
        ServerModelSegmentFitIns
            The instructions for the next round of training. TODO: as of now, all the clients
            interact with the same initial version of the server-model segment, though it might
            make sense to consider giving more flexibility
        """

    @abstractmethod
    def configure_client_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the client-side model segments for the next round of evaluation

        Parameters
        ----------
        server_round : int
            The current round of split learning
        parameters : Parameters
            The current version of the client-side model parameters
        client_manager : ClientManager
            The client manager which holds all currently connected clients

        Returns
        -------
        List[Tuple[ClientProxy, EvaluateIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of split learning evaluation.
        """

    @abstractmethod
    def configure_server_evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> ServerModelSegmentEvaluateIns:
        """Configure the server-side model segment for the next round of evaluation

        Parameters
        ----------
        server_round : int
            The current round of split learning
        parameters : Parameters
            The current version of the server-side model parameters

        Returns
        -------
        ServerModelSegmentEvaluateIns
            The configuration for the next round of model evaluation
        """

    @abstractmethod
    def aggregate_client_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate the client-side training results.

        Parameters
        ----------
        server_round : int
            The current round of split learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        parameters : Tuple[Optional[Parameters], Dict[str, Scalar]]
            If parameters are returned, then the server will treat these as the
            new global client-side model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """

    @abstractmethod
    def aggregate_server_fit(
        self,
        server_round: int,
        results: List[ServerModelSegmentFitRes]
    ) -> Optional[Parameters]:
        """Aggregate the server-side training results

        Parameters
        ----------
        server_round : int
            The current round of split learning
        results : List[ServerModelSegmentFitRes]
            Successful updates from previously active server model segments

        Returns
        -------
        Optional[Parameters]
            New version of the server-side model parameters. If None is returned, the
            server-side model parameters are not updated
        """

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ):
        """Aggregate evaluation results.

        Parameters
        ----------
        server_round : int
            The current round of split learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured clients.
            Each pair of `(ClientProxy, FitRes` constitutes a successful update from
            one of the previously selected clients. Not that not all previously selected
            clients are necessarily included in this list: a client might drop out
            and not submit a result. For each client that did not submit an update,
            there should be an `Exception` in `failures`.
        failures : List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
            Exceptions that occurred while the server was waiting for client updates.

        Returns
        -------
        aggregation_result : Tuple[Optional[float], Dict[str, Scalar]]
            The aggregated evaluation result. Aggregation typically uses some variant
            of a weighted average.
        """
