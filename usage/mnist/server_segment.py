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

from .models import ServerModel
from .common import get_parameters, set_parameters, seed


class SimpleServerModelSegment(ServerModelSegment):

    def __init__(self) -> None:
        seed()
        self.model = ServerModel()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)

    def serve_prediction_request(self, batch_data: BatchPredictionIns) -> BatchPredictionRes:
        embeddings = bytes_to_torch(batch_data.embeddings, False)
        with torch.no_grad():
           preds = self.model(embeddings)
        preds = torch.argmax(preds, axis=1)
        preds = torch_to_bytes(preds)
        return BatchPredictionRes(preds)

    def serve_gradient_update_request(
        self,
        batch_data: GradientDescentDataBatchIns
    ) -> GradientDescentDataBatchRes:
        embeddings = bytes_to_torch(batch_data.embeddings, True)
        labels = bytes_to_torch(batch_data.labels, False)
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

    def configure_fit(self, ins: ServerModelSegmentFitIns) -> bool:
        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))
        return True

    def configure_evaluate(self, ins: ServerModelSegmentEvaluateIns) -> bool:
        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))
        return True
