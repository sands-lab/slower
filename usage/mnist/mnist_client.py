import time

import torch
from flwr.common import (
    GetParametersIns,
    FitIns,
    GetParametersRes,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Status,
    Code
)

from slower.server.server_model_segment.proxy.server_model_segment_proxy import ServerModelSegmentProxy
from slower.client.client import Client
from slower.common import (
    BatchPredictionIns,
    GradientDescentDataBatchIns,
    torch_to_bytes,
    bytes_to_torch
)

from usage.common.helper import get_parameters, set_parameters, seed
from usage.mnist.models import ClientModel
from usage.mnist.common import get_dataloader


class MnistClient(Client):

    def __init__(self, cid) -> None:
        super().__init__()
        seed()
        self.cid = cid
        self.model = ClientModel()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(get_parameters(self.model))
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )

    def fit(self, ins: FitIns, server_model_segment_proxy: ServerModelSegmentProxy) -> FitRes:
        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        dataloader = get_dataloader()
        times = []
        for images, labels in dataloader:
            embeddings = self.model(images)
            s = time.time()
            compute_error_ins = GradientDescentDataBatchIns(
                embeddings=torch_to_bytes(embeddings),
                labels=torch_to_bytes(labels)
            )
            times.append(time.time() - s)
            error = server_model_segment_proxy.serve_gradient_update_request(compute_error_ins, None)
            s = time.time()
            error = bytes_to_torch(error.gradient, False)
            times.append(time.time() - s)

            self.model.zero_grad()
            embeddings.backward(error)
            optimizer.step()

        print(sum(times))
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(get_parameters(self.model)),
            num_examples=len(dataloader.dataset),
            metrics={}
        )


    def evaluate(self, ins: EvaluateIns, server_model_segment_proxy: ServerModelSegmentProxy) -> EvaluateRes:
        dataloader = get_dataloader()
        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))
        correct = 0
        times = []
        for images, labels in dataloader:
            embeddings = self.model(images)
            s = time.time()
            ins = BatchPredictionIns(embeddings=torch_to_bytes(embeddings))
            times.append(time.time() - s)
            preds = server_model_segment_proxy.serve_prediction_request(ins, timeout=None)
            s = time.time()
            preds = bytes_to_torch(preds.predictions, False).int()
            times.append(time.time() - s)

            correct += (preds == labels).int().sum()
        print(sum(times))
        accuracy = float(correct / len(dataloader.dataset))
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=accuracy,
            num_examples=len(dataloader.dataset),
            metrics={}
        )
