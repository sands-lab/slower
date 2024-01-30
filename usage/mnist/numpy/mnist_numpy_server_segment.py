import torch


from flwr.common import GetParametersRes

from slower.common import (
    torch_to_bytes,
    bytes_to_torch
)
from slower.server.server_model_segment.numpy_server_model_segment import NumPyServerModelSegment

from usage.mnist.models import ServerModel
from usage.common.helper import get_parameters, set_parameters


class MnistNumpyServerSegment(NumPyServerModelSegment):

    def __init__(self) -> None:
        self.model = ServerModel()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)

    def serve_prediction_request(self, embeddings) -> bytes:
        embeddings = bytes_to_torch(embeddings, False)
        with torch.no_grad():
           preds = self.model(embeddings)
        preds = torch.argmax(preds, axis=1)
        preds = torch_to_bytes(preds)
        return preds

    def serve_gradient_update_request(self, embeddings, labels) -> bytes:
        embeddings = bytes_to_torch(embeddings, True)
        labels = bytes_to_torch(labels, False)
        preds = self.model(embeddings)
        loss = self.criterion(preds, labels)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        error = torch_to_bytes(embeddings.grad)
        return error

    def get_parameters(self) -> GetParametersRes:
        parameters = get_parameters(self.model)
        return parameters

    def configure_fit(self, parameters, config):
        set_parameters(self.model, parameters)

    def configure_evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
