import torch


from flwr.common import GetParametersRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import Code

from slower.common import (
    BatchPredictionIns,
    BatchPredictionRes,
    GradientDescentDataBatchIns,
    GradientDescentDataBatchRes,
    ServerModelSegmentEvaluateIns,
    ServerModelSegmentFitIns,
    torch_to_bytes,
    bytes_to_torch
)
from slower.server.server_model_segment.server_model_segment import ServerModelSegment

from usage.cifar10.models import get_server_model
import usage.cifar10.constants as constants

from usage.common.helper import seed, set_parameters, get_parameters


class CifarRawServerSegment(ServerModelSegment):

    def __init__(self) -> None:
        seed()
        self.model = get_server_model(constants.N_CLIENT_LAYERS)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)

    def serve_prediction_request(self, batch: BatchPredictionIns) -> BatchPredictionRes:
        embeddings = bytes_to_torch(batch.embeddings, False)
        with torch.no_grad():
            preds = self.model(embeddings)
        preds = torch.argmax(preds, axis=1)
        preds = torch_to_bytes(preds)
        return BatchPredictionRes(preds)

    def serve_gradient_update_request(
        self,
        batch: GradientDescentDataBatchIns
    ) -> GradientDescentDataBatchRes:
        embeddings = bytes_to_torch(batch.embeddings, True)
        labels = bytes_to_torch(batch.labels, False)
        preds = self.model(embeddings)
        loss = self.criterion(preds, labels)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        error = embeddings.grad
        return GradientDescentDataBatchRes(gradient=torch_to_bytes(error))

    def get_parameters(self) -> GetParametersRes:
        parameters = ndarrays_to_parameters(get_parameters(self.model))
        return GetParametersRes(
            status=Code.OK,
            parameters=parameters
        )

    def configure_fit(self, ins: ServerModelSegmentFitIns):
        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))

    def configure_evaluate(self, ins: ServerModelSegmentEvaluateIns):
        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))
