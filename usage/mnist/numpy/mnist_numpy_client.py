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

from usage.common.helper import (
    get_parameters,
    set_parameters
)
from usage.mnist.models import ClientModel
from usage.mnist.common import get_dataloader


class MnistNumpyClient(NumPyClient):

    def __init__(self, cid) -> None:
        super().__init__()
        self.model = ClientModel()
        self.cid = cid

    def get_parameters(self, config) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self,
        parameters,
        server_model_segment_proxy: ServerModelSegmentProxy,
        config,
    ) -> FitRes:
        _ = (config,)
        set_parameters(self.model, parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        dataloader = get_dataloader()

        for images, labels in dataloader:
            embeddings = self.model(images)

            error = server_model_segment_proxy.serve_gradient_update_request_wrapper(
                embeddings=torch_to_bytes(embeddings),
                labels=torch_to_bytes(labels),
                timeout=None
            )
            error = bytes_to_torch(error, False)

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
        dataloader = get_dataloader()
        set_parameters(self.model, parameters)

        correct = 0
        for images, labels in dataloader:
            embeddings = self.model(images)
            preds = server_model_segment_proxy.serve_prediction_request_wrapper(
                embeddings=torch_to_bytes(embeddings),
                timeout=None
            )
            preds = bytes_to_torch(preds, False).int()
            correct += (preds == labels).int().sum()

        accuracy = float(correct / len(dataloader.dataset))
        return accuracy, len(dataloader.dataset), {}
