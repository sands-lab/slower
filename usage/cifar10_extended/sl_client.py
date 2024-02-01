import torch
from flwr.common import (
    FitRes,
    EvaluateRes,
    NDArrays,
)

from slower.server.server_model_segment.proxy.server_model_segment_proxy import (
    ServerModelSegmentProxy
)
from slower.client.numpy_client import NumPyClient
from slower.common import (
    torch_to_bytes,
    bytes_to_torch
)

from usage.cifar10_extended.models import get_client_model
from usage.common.helper import (
    get_parameters,
    set_parameters
)
from usage.cifar10_extended.common import get_dataloader
import usage.cifar10_extended.constants as constants


class CifarSlClient(NumPyClient):

    def __init__(self, cid) -> None:
        super().__init__()
        self.cid = cid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_client_model(constants.N_CLIENT_LAYERS).to(self.device)

    def get_parameters(self, config) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self,
        parameters,
        server_model_segment_proxy: ServerModelSegmentProxy,
        config,
    ) -> FitRes:
        print(f"Fitting client {self.cid}")
        set_parameters(self.model, parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"])
        dataloader = get_dataloader(True, self.cid)

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            embeddings = self.model(images)

            error = server_model_segment_proxy.serve_gradient_update_request_wrapper(
                embeddings=torch_to_bytes(embeddings),
                labels=torch_to_bytes(labels),
                timeout=None
            )
            error = bytes_to_torch(error, False)
            error = error.to(self.device)

            self.model.zero_grad()
            embeddings.backward(error)
            optimizer.step()

        return get_parameters(self.model), len(dataloader.dataset), {}

    def evaluate(
        self,
        parameters,
        server_model_segment_proxy: ServerModelSegmentProxy,
        config
    ) -> EvaluateRes:
        dataloader = get_dataloader(False, self.cid)
        set_parameters(self.model, parameters)

        correct = 0
        for images, labels in dataloader:
            images = images.to(self.device)

            embeddings = self.model(images)
            preds = server_model_segment_proxy.serve_prediction_request_wrapper(
                embeddings=torch_to_bytes(embeddings),
                timeout=None
            )
            preds = bytes_to_torch(preds, False).int()
            correct += (preds == labels).int().sum()

        accuracy = float(correct / len(dataloader.dataset))
        return accuracy, len(dataloader.dataset), {}
