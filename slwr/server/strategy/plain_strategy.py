import uuid
import copy
from logging import ERROR
from typing import Optional, List, Tuple, Union, Dict, Callable

from flwr.common import (
    Parameters,
    FitIns,
    EvaluateIns,
    FitRes,
    Scalar,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    log,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy

from slwr.server.server_model.server_model import ServerModel
from slwr.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)
from slwr.server.strategy import Strategy
from slwr.server.server_model.utils import ClientRequestGroup


class PlainSlStrategy(Strategy):

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        init_server_model_fn: Callable[[], ServerModel],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        config_server_segnent_fn: Optional[Callable[[str], Dict[str, Scalar]]] = None,
        config_client_fit_fn: Optional[Callable[[str], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn = None,
        fit_metrics_aggregation_fn = None,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        common_server_model: bool = True,
        process_clients_as_batch: bool = False,
    ) -> None:
        super().__init__()
        self.init_server_model_fn = init_server_model_fn
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.config_server_segnent_fn = config_server_segnent_fn
        self.config_client_fit_fn = config_client_fit_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self._cid_to_sid_mapping = {}
        self.common_server_model = common_server_model
        self.process_clients_as_batch = process_clients_as_batch
        self._round_active_clients = []
        self.requests_state: Dict[Tuple[str, str], Dict[str, ClientRequestGroup]] = {}

    def init_server_model_fn(self) -> ServerModel:
        return self.init_server_model_fn().to_server_model()

    def evaluate(
        self,
        server_round: int,
        client_parameters: Parameters,
        server_parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Distributes to all the sampled clients the current parameters and the configuration
        obtained via self.configure_client_fit_fn

        Parameters
        ----------
        server_round : int
            Current training epoch
        parameters : Parameters
            Current parameters of the client-side model
        client_manager : ClientManager
            the client manager

        Returns
        -------
        List[Tuple[ClientProxy, FitIns]]
            Instructions for the client
        """
        config = {}
        if self.config_client_fit_fn:
            config = self.config_client_fit_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        self._round_active_clients = [client.cid for client in clients]

        # Return client/config pairs
        return [(client, copy.deepcopy(fit_ins)) for client in clients]

    def configure_server_fit(
        self,
        server_round: int,
        parameters: NDArrays,
        cids: List[str],
    ) -> List[ServerModelFitIns]:
        """Distribute to all the server models the current weights of the server model

        Parameters
        ----------
        server_round : int
            Training epoch
        parameters : NDArrays
            Current version of the server model weights
        cids : List[str]
            List of client IDs selected for this round of training

        Returns
        -------
        List[ServerModelFitIns]
            A set of configurations of the N server models. By default the number of server models
            is set to 1. If want to get a different number of server models, make the
            `config_server_segment_fn` return a dictionary with the key `num_server_models`
            that states how many server models to be used
        """
        _ = (server_round, )
        config = {}
        if self.config_server_segnent_fn:
            config = self.config_server_segnent_fn(server_round, cids)
        num_server_models = 1 if self.common_server_model else len(cids)
        if num_server_models == 1:
            self._cid_to_sid_mapping = {cid: "" for cid in cids}
            return [
                ServerModelFitIns(parameters=parameters, config=config, sid="")
            ]
        self._cid_to_sid_mapping = {cid: cid for cid in cids}
        return [
            ServerModelFitIns(parameters=parameters, config=config, sid=cid)
            for cid in cids
        ]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        evaluate_ins = EvaluateIns(parameters, {})

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        self._round_active_clients = [client.cid for client in clients]

        # Return client/config pairs
        return [(client, copy.deepcopy(evaluate_ins)) for client in clients]

    def configure_server_evaluate(
        self,
        server_round: int,
        parameters: NDArrays,
        cids: List[str],
    ) -> List[ServerModelEvaluateIns]:
        # for simplicity assume that the server model is the same for all clients
        self._cid_to_sid_mapping = {cid: "" for cid in cids}
        return [ServerModelEvaluateIns(parameters=parameters, config={}, sid="")]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        _ = (server_round, failures,)
        self.requests_state = {}
        self._round_active_clients = []
        for failure in failures:
            log(ERROR, str(failure))

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        return parameters_aggregated, aggregated_metrics

    def aggregate_server_fit(
        self,
        server_round: int,
        results: List[ServerModelFitRes]
    ) -> Optional[NDArrays]:
        """Does a weighted average of the received server model weights. It weights the
        server parameters by a parameter `num_examples` provided in the config of every
        server result (if not preset, `num_examples` is set to 1, hence performing a plain average)

        Parameters
        ----------
        server_round : int
            Training epoch
        results : List[ServerModelFitRes]
            Updated server model weights and corresponding configuration for aggregating weights

        Returns
        -------
        Optional[NDArrays]
            Updated server model weights
        """
        _ = (server_round, )
        weights_results = [
            (res.parameters, res.config.pop("num_examples", 1))
            for res in results
        ]
        parameters_aggregated = aggregate(weights_results)

        return parameters_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ):
        self.requests_state = {}
        _ = (server_round, failures, )
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        aggregated_metrics = {}
        if self.evaluate_metrics_aggregation_fn is not None:
            evaluation_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(evaluation_metrics)

        self._round_active_clients = []
        return loss_aggregated, aggregated_metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def route_client_request(
        self,
        cid: str,
        method_name: str,
    ) -> Tuple[ClientRequestGroup, Optional[Callable]]:
        sid = self._cid_to_sid_mapping[cid]
        request_key = (sid, method_name)
        if request_key not in self.requests_state:
            self.requests_state[request_key] = {}
        for reqg in self.requests_state[request_key].values():
            if not reqg.is_ready():
                return reqg, lambda: None

        # all requests are ready. Create a new one
        request_group = ClientRequestGroup(sid)
        request_id = str(uuid.uuid4())[:8]
        self.requests_state[request_key][request_id] = request_group
        callback_fn = lambda: self.requests_state[request_key].pop(request_id)
        return request_group, callback_fn

    def mark_ready_requests(self):
        for reqg_dict in self.requests_state.values():
            for reqg in reqg_dict.values():
                if (
                    not self.process_clients_as_batch or # process batch individually
                    reqg.get_num_batches() == len(self._round_active_clients)
                ):
                    # each client is processed individually
                    reqg.mark_as_ready()

    def mark_client_as_done(self, cid):
        self._round_active_clients.remove(cid)
        self.mark_ready_requests()
