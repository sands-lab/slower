import timeit
from logging import DEBUG, INFO
from typing import Dict, Optional, Tuple, List

from flwr.common import (
    Parameters,
    Scalar,
    log,
    NDArrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import (
    fit_clients,
    evaluate_clients,
    FitResultsAndFailures,
    EvaluateResultsAndFailures,
    Server as FlwrServer
)

from slwr.server.strategy import Strategy
from slwr.server.server_model.manager.manager import ServerModelManager
from slwr.common import ServerModelFitIns
from slwr.common.constants import CLIENT_ID_CONFIG_KEY


class Server(FlwrServer):
    """Flower server."""

    def __init__(
        self,
        *,
        server_model_manager: ServerModelManager,
        client_manager: ClientManager,
        strategy: Strategy,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        # rename client parameters from `parameters` to `client_parameters`
        self.client_parameters = self.parameters
        del self.parameters

        self.server_model_manager: ServerModelManager = server_model_manager
        self.server_model_manager.set_strategy(strategy)
        self.server_parameters: NDArrays = []

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.client_parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        self.server_parameters = self._get_initial_server_parameters()

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                client_parameters_prime, server_parameters_prime, fit_metrics, _ = res_fit
                if client_parameters_prime:
                    self.client_parameters = client_parameters_prime
                if server_parameters_prime:
                    self.server_parameters = server_parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(
                current_round,
                client_parameters=self.client_parameters,
                server_parameters=self.server_parameters
            )
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history, elapsed

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.client_parameters,
            client_manager=self._client_manager,
        )
        for proxy, ins in client_instructions:
            ins.config[CLIENT_ID_CONFIG_KEY] = proxy.cid

        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        server_model_config = self.strategy.configure_server_evaluate(
            server_round=server_round,
            parameters=self.server_parameters,
            cids=[ins[0].cid for ins in client_instructions]
        )
        self.server_model_manager.initialize_server_models(server_model_config)

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        self.server_model_manager.end_round()
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.client_parameters,
            client_manager=self._client_manager,
        )
        for proxy, ins in client_instructions:
            ins.config[CLIENT_ID_CONFIG_KEY] = proxy.cid

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # start the server proxies
        server_model_config: List[ServerModelFitIns] = self.strategy.configure_server_fit(
            server_round=server_round,
            parameters=self.server_parameters,
            cids=[proxy.cid for proxy, _ in client_instructions]
        )
        self.server_model_manager.initialize_server_models(server_model_config)

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        server_fit_res = self.server_model_manager.collect_server_fit_results()

        # Aggregate training results
        aggregated_client_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)
        server_parameters_aggregated = \
            self.strategy.aggregate_server_fit(server_round, server_fit_res)

        client_parameters_aggregated, metrics_aggregated = aggregated_client_result
        return (
            client_parameters_aggregated,
            server_parameters_aggregated,
            metrics_aggregated,
            (results, failures)
        )

    def _get_initial_server_parameters(self) -> Parameters:
        parameters: Optional[Parameters] = self.strategy.initialize_server_parameters()
        if parameters is not None:
            log(INFO, "Using initial server parameters provided by the strategy")
            return parameters

        # initialize one random server trainer and get parameters from it
        # it's lightweight operation so we perform it in the main process
        log(INFO, "Initializing random server model in order to fetch initial parameters")
        server_model = \
            self.strategy.init_server_model_fn().to_server_model()
        get_parameters_res = server_model.get_parameters()
        log(INFO, "Received initial parameters from a virtual server model")
        return get_parameters_res
