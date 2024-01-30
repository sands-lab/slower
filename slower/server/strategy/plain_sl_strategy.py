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

from slower.client.proxy.client_proxy import ClientProxy
from slower.server.server_model_segment.server_model_segment import ServerModelSegment
from slower.common import (
    ServerModelSegmentEvaluateIns,
    ServerModelSegmentFitIns,
    ServerModelSegmentFitRes,
)

from slower.server.strategy import SlStrategy


class PlainSlStrategy(SlStrategy):

    def __init__(
            self,
            *,
            common_server: bool,
            init_server_model_segment_fn: Callable[[], ServerModelSegment],
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0) -> None:
        super().__init__()
        self._common_server = common_server
        self.init_server_model_segment_fn = init_server_model_segment_fn
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate

    def has_common_server_model_segment(self) -> bool:
        return self._common_server

    def init_server_model_segment_fn(self) -> ServerModelSegment:
        return self.init_server_model_segment_fn().to_server_model_segment()

    def evaluate(
        self,
        server_round: int,
        client_parameters: Parameters,
        server_parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

    def initialize_client_parameters(
        self,
        client_manager: ClientManager
    ) -> Optional[Parameters]:
        return None

    def initialize_server_parameters(
        self
    ) -> Optional[Parameters]:
        return None

    def configure_client_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        fit_ins = FitIns(parameters, {})

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
    ) -> ServerModelSegmentFitIns:
        return ServerModelSegmentFitIns(parameters=parameters, config={})

    def configure_client_evaluate(
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
    ) -> ServerModelSegmentEvaluateIns:
        return ServerModelSegmentEvaluateIns(parameters=parameters, config={})

    def aggregate_client_fit(
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

        return parameters_aggregated, {}

    def aggregate_server_fit(
        self,
        server_round: int,
        results: List[ServerModelSegmentFitRes]
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
        return loss_aggregated, {}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, 1), 1

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, 1), 1
