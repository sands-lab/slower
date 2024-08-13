from typing import Optional, List, Tuple, Union, Dict, Callable

from flwr.common import (
    Parameters,
    FitIns,
    EvaluateIns,
    FitRes,
    Scalar,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy

from slower.server.server_model.server_model import ServerModel
from slower.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)
from slower.server.strategy import SlStrategy


class PlainSlStrategy(SlStrategy):

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

    def init_server_model_fn(self) -> ServerModel:
        return self.init_server_model_fn().to_server_model()

    def evaluate(
        self,
        server_round: int,
        client_parameters: Parameters,
        server_parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

    def initialize_parameters(
        self,
        client_manager: ClientManager
    ) -> Optional[Parameters]:
        client_manager.wait_for(self.min_available_clients)
        return None

    def initialize_server_parameters(
        self
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
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

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_server_fit(
        self,
        server_round: int,
        parameters: Parameters
    ) -> ServerModelFitIns:
        config = {}
        if self.config_server_segnent_fn:
            config = self.config_server_segnent_fn(server_round)
        return ServerModelFitIns(parameters=parameters, config=config)

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        evaluate_ins = EvaluateIns(parameters, {})

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def configure_server_evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> ServerModelEvaluateIns:
        return ServerModelEvaluateIns(parameters=parameters, config={})

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
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
    ) -> Optional[Parameters]:
        weights_results = [
            (parameters_to_ndarrays(res.parameters), res.num_examples)
            for res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        return parameters_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ):
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

        return loss_aggregated, aggregated_metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
