import torch

from flwr.common import GetParametersRes

from slower.common import (
    torch_to_bytes,
    bytes_to_torch
)
from slower.server.server_model_segment.numpy_server_model_segment import NumPyServerModelSegment

from usage.cifar10_extended.models import get_server_model
from usage.common.helper import get_parameters, set_parameters
import usage.cifar10_extended.constants as constants


class CifarSlServerSegment(NumPyServerModelSegment):

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_server_model(constants.N_CLIENT_LAYERS).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def serve_prediction_request(self, embeddings) -> bytes:
        embeddings = bytes_to_torch(embeddings, False).to(self.device)
        with torch.no_grad():
            preds = self.model(embeddings)
        preds = torch.argmax(preds, axis=1)
        preds = torch_to_bytes(preds)
        return preds

    def serve_gradient_update_request(self, embeddings, labels) -> bytes:
        # NOTE: the deserialization of the embeddings vector needs to be the following
        # setting first grad and then moving the tensor to CUDA will result in an error
        # not completely sure why this happens...
        # see https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751
        # for a starting point...
        embeddings = bytes_to_torch(embeddings, False)
        embeddings = embeddings.to(self.device)
        embeddings.requires_grad_(True)
        labels = bytes_to_torch(labels, False).to(self.device)
        preds = self.model(embeddings)
        loss = self.criterion(preds, labels)
        assert bool(embeddings.requires_grad)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        error = torch_to_bytes(embeddings.grad)
        return error

    def get_parameters(self) -> GetParametersRes:
        parameters = get_parameters(self.model)
        return parameters

    def configure_fit(self, parameters, config):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"])
        set_parameters(self.model, parameters)
        self.model.train()

    def configure_evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
