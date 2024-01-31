from abc import ABC, abstractmethod
from typing import Optional

from flwr.common import (
    GetParametersIns,
    FitIns,
    GetParametersRes,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ReconnectIns,
)


class ClientProxy(ABC):

    node_id: int

    def __init__(self, cid: str) -> None:
        self.cid = cid

    @abstractmethod
    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float]
    ) -> GetParametersRes:
        """Return the current client-side model parameters"""


    @abstractmethod
    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float]
    ) -> FitRes:
        """Refine the provided client-side model parameters by training on the local dataset
        with the collaboration of the server-side segment of the model"""

    @abstractmethod
    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float]
    ) -> EvaluateRes:
        """Test the provided client-side model parameters by evaluating them on
        the local dataset with the collaboration of the server-side segment of the model"""

    @abstractmethod
    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float]
    ):
        """Disconnect and (optionally) reconnect later."""
