from abc import ABC

from flwr.client.workload_state import WorkloadState
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)

from slower.server.server_model_segment.proxy.server_model_segment_proxy import (
    ServerModelSegmentProxy
)


class Client(ABC):

    state: WorkloadState

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the current parameters of the client-side segment of the model

        Parameters
        ----------
        ins : GetParametersIns
            The get parameters instructions received from the server containing
            a dictionary of configuration values.

        Returns
        -------
        GetParametersRes
            The current local model parameters.
        """
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.GET_PARAMETERS_NOT_IMPLEMENTED,
                message="Client does not implement `get_parameters`",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )


    # pylint: disable=unused-argument
    def fit(
            self,
            ins: FitIns,
            server_model_segment_proxy: ServerModelSegmentProxy
        ) -> FitRes:
        """Refine the client-side model segment using the locally held dataset by collaborating
        with the server-side segment of the model

        Parameters
        ----------
        ins : FitIns
            The training instructions containing the current version of the
            globally available client-side segment of the model, and a dictionary
            of configuration values used to customize the local training process.
        server_model_segment_proxy: ServerModelSegmentProxy
            Interface to the server-side segment of the model. Within the `fit` function,
            the `serve_gradient_update_request` method should be used
        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """
        _ = (self, ins)
        return FitRes(
            status=Status(
                code=Code.FIT_NOT_IMPLEMENTED,
                message="Client does not implement `fit`",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=0,
            metrics={},
        )

    # pylint: disable=unused-argument
    def evaluate(
            self,
            ins: EvaluateIns,
            server_model_segment_proxy: ServerModelSegmentProxy
        ) -> EvaluateRes:
        """Evaluate the provided client-side segment model parameters by collaborating with the
        server side segment of the model

        Parameters
        ----------
        ins : EvaluateIns
            The evaluation instructions containing (global) model parameters of the
            client-side segment of the model received from the server and a dictionary
            of configuration values used to customize the local evaluation process.
        server_model_segment_proxy: ServerModelSegmentProxy
            Interface to the server-side segment of the model. Within the `evaluate` function,
            the `serve_prediction_request` method should be used

        Returns
        -------
        EvaluateRes
            The evaluation result containing the loss on the local dataset and
            other details such as the number of local data examples used for
            evaluation.
        """
        _ = (self, ins)
        return EvaluateRes(
            status=Status(
                code=Code.EVALUATE_NOT_IMPLEMENTED,
                message="Client does not implement `evaluate`",
            ),
            loss=0.0,
            num_examples=0,
            metrics={},
        )

    def get_state(self) -> WorkloadState:
        """Get the run state from this client."""
        return self.state

    def set_state(self, state: WorkloadState) -> None:
        """Apply a run state to this client."""
        self.state = state

    def to_client(self):
        return self


# The following functions are directly copied from flower
def has_get_parameters(client: Client) -> bool:
    """Check if Client implements get_parameters."""
    return type(client).get_parameters != Client.get_parameters


def has_fit(client: Client) -> bool:
    """Check if Client implements fit."""
    return type(client).fit != Client.fit


def has_evaluate(client: Client) -> bool:
    """Check if Client implements evaluate."""
    return type(client).evaluate != Client.evaluate


def maybe_call_get_parameters(
    client: Client, get_parameters_ins: GetParametersIns
) -> GetParametersRes:
    """Call `get_parameters` if the client overrides it."""
    # Check if client overrides `get_parameters`
    if not has_get_parameters(client=client):
        # If client does not override `get_parameters`, don't call it
        status = Status(
            code=Code.GET_PARAMETERS_NOT_IMPLEMENTED,
            message="Client does not implement `get_parameters`",
        )
        return GetParametersRes(
            status=status,
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    # If the client implements `get_parameters`, call it
    return client.get_parameters(get_parameters_ins)


def maybe_call_fit(
    client: Client,
    fit_ins: FitIns,
    server_model_segment_proxy: ServerModelSegmentProxy,
) -> FitRes:
    """Call `fit` if the client overrides it."""
    # Check if client overrides `fit`
    if not has_fit(client=client):
        # If client does not override `fit`, don't call it
        status = Status(
            code=Code.FIT_NOT_IMPLEMENTED,
            message="Client does not implement `fit`",
        )
        return FitRes(
            status=status,
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=0,
            metrics={},
        )

    # If the client implements `fit`, call it
    return client.fit(fit_ins, server_model_segment_proxy)


def maybe_call_evaluate(
    client: Client,
    evaluate_ins: EvaluateIns,
    server_model_segment_proxy: ServerModelSegmentProxy
) -> EvaluateRes:
    """Call `evaluate` if the client overrides it."""
    # Check if client overrides `evaluate`
    if not has_evaluate(client=client):
        # If client does not override `evaluate`, don't call it
        status = Status(
            code=Code.EVALUATE_NOT_IMPLEMENTED,
            message="Client does not implement `evaluate`",
        )
        return EvaluateRes(
            status=status,
            loss=0.0,
            num_examples=0,
            metrics={},
        )

    # If the client implements `evaluate`, call it
    return client.evaluate(evaluate_ins, server_model_segment_proxy)
