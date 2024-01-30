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

from usage.cifar10.common import get_dataloader
from usage.cifar10.models import get_client_model
from usage.cifar10.constants import N_CLIENT_LAYERS

from usage.common.helper import (
    seed,
    set_parameters,
    get_parameters,
    ExecutionTime,
    print_metrics,
    init_complete_metrics_dict
)


class CifarRawClient(Client):

    def __init__(self, cid) -> None:
        super().__init__()
        seed()
        self.cid = cid
        self.model = get_client_model(N_CLIENT_LAYERS)

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

        tot_zeros, tot_sent = 0, 0
        times = init_complete_metrics_dict()

        begin = time.time()
        for images, labels in dataloader:
            with ExecutionTime(times["forward"]):
                embeddings = self.model(images)
                tot_zeros += (embeddings < 1e-9).int().sum()
                tot_sent += embeddings.numel()
            with ExecutionTime(times["serialization"]):
                compute_error_ins = GradientDescentDataBatchIns(
                    embeddings=torch_to_bytes(embeddings),
                    labels=torch_to_bytes(labels)
                )
            with ExecutionTime(times["communication"]):
                error = server_model_segment_proxy.serve_gradient_update_request(compute_error_ins, None)
            with ExecutionTime(times["serialization"]):

                error = bytes_to_torch(error.gradient, False)
            with ExecutionTime(times["backward"]):
                self.model.zero_grad()
                embeddings.backward(error)
                optimizer.step()

        tmp = {
            "% of sent zeros": {tot_zeros / tot_sent},
            "total time": time.time() - begin
        }
        print_metrics(True, {
            **times,
            **tmp
        })
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
        serialization_times = []
        for images, labels in dataloader:
            embeddings = self.model(images)
            with ExecutionTime(serialization_times):
                ins = BatchPredictionIns(embeddings=torch_to_bytes(embeddings))
            preds = server_model_segment_proxy.serve_prediction_request(ins, timeout=None)
            with ExecutionTime(serialization_times):

                preds = bytes_to_torch(preds.predictions, False).int()

            correct += (preds == labels).int().sum()
        print(f"Evaluation - serialization time: {sum(serialization_times)}")
        accuracy = float(correct / len(dataloader.dataset))
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=accuracy,
            num_examples=len(dataloader.dataset),
            metrics={}
        )
