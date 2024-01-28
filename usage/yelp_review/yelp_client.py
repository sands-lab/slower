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
    torch_list_to_bytes,
    bytes_to_torch_list,
    torch_to_bytes,
    bytes_to_torch
)

from common import get_dataloader
from models import ClientBert
from constants import N_CLIENT_LAYERS
from compression import compress_embeddings, uncompress_embeddings

from usage.common.helper import seed, set_parameters, get_parameters


class YelpClient(Client):

    def __init__(self, cid) -> None:
        super().__init__()
        seed()
        self.cid = cid
        self.model = ClientBert(N_CLIENT_LAYERS)
        self.model.eval()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(get_parameters(self.model))
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )

    def fit(self, ins: FitIns, server_model_segment_proxy: ServerModelSegmentProxy) -> FitRes:
        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        dataloader = get_dataloader()
        times = []
        for batch in dataloader:
            embeddings = self.model(**{k: v for k, v in batch.items() if k != "labels"})

            s = time.time()
            compressed_embs = compress_embeddings(embeddings, batch["attention_mask"].sum(axis=1))
            compute_error_ins = GradientDescentDataBatchIns(
                embeddings=torch_list_to_bytes(compressed_embs),
                labels=torch_to_bytes(batch["labels"])
            )
            times.append(time.time() - s)

            error = server_model_segment_proxy.serve_gradient_update_request(compute_error_ins, None)

            s = time.time()
            error = bytes_to_torch_list(error.gradient)
            error, _ = uncompress_embeddings(error)
            times.append(time.time() - s)

            self.model.zero_grad()
            embeddings.backward(error)
            optimizer.step()

        print(f"Train: compression + serialization: {sum(times)}")
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
        for batch in dataloader:
            embeddings = self.model(**{k: v for k, v in batch.items() if k != "labels"})

            s = time.time()
            compressed_embs = compress_embeddings(embeddings, batch["attention_mask"].sum(axis=1))
            ins = BatchPredictionIns(embeddings=torch_list_to_bytes(compressed_embs))
            times.append(time.time() - s)

            preds = server_model_segment_proxy.serve_prediction_request(ins, timeout=None)

            s = time.time()
            preds = bytes_to_torch(preds.predictions, False).int()
            times.append(time.time() - s)

            correct += (preds == batch["labels"]).int().sum()
        print(f"Evaluate: compression + serialization: {sum(times)}")

        accuracy = float(correct / len(dataloader.dataset))
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=accuracy,
            num_examples=len(dataloader.dataset),
            metrics={}
        )
