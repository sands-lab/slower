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

from usage.yelp_review.common import get_dataloader
from usage.yelp_review.models import ClientBert
from usage.yelp_review.constants import N_CLIENT_LAYERS
from usage.yelp_review.compression import compress_embeddings, uncompress_embeddings

from usage.common.helper import (
    seed,
    set_parameters,
    get_parameters,
    init_complete_metrics_dict,
    ExecutionTime,
    print_metrics
)


class YelpRawClient(Client):

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
        tot_zeros, tot_sent = 0, 0
        times = init_complete_metrics_dict()
        for batch in dataloader:
            with ExecutionTime(times["forward"]):
                embeddings = self.model(**{k: v for k, v in batch.items() if k != "labels"})

            with ExecutionTime(times["serialization"]):
                compressed_embs = compress_embeddings(embeddings, batch["attention_mask"].sum(axis=1))
                tot_zeros += sum((e < 1e-9).int().sum() for e in compressed_embs)
                tot_sent += sum(e.numel() for e in compressed_embs)
                compute_error_ins = GradientDescentDataBatchIns(
                    embeddings=torch_list_to_bytes(compressed_embs),
                    labels=torch_to_bytes(batch["labels"])
                )

            with ExecutionTime(times["communication"]):
                error = server_model_segment_proxy.serve_gradient_update_request(compute_error_ins, None)

            with ExecutionTime(times["serialization"]):
                error = bytes_to_torch_list(error.gradient)
                error, _ = uncompress_embeddings(error)

            with ExecutionTime(times["backward"]):
                self.model.zero_grad()
                embeddings.backward(error)
                optimizer.step()

        print_metrics(True, {
            **times,
            **{"Percentage of zeros": tot_zeros / tot_sent}
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

        for batch in dataloader:
            embeddings = self.model(**{k: v for k, v in batch.items() if k != "labels"})

            with ExecutionTime(serialization_times):
                compressed_embs = compress_embeddings(embeddings, batch["attention_mask"].sum(axis=1))
                ins = BatchPredictionIns(embeddings=torch_list_to_bytes(compressed_embs))

            preds = server_model_segment_proxy.serve_prediction_request(ins, timeout=None)

            with ExecutionTime(serialization_times):
                preds = bytes_to_torch(preds.predictions, False).int()

            correct += (preds == batch["labels"]).int().sum()
        print(f"Evaluate: compression + serialization: {sum(serialization_times)}")

        accuracy = float(correct / len(dataloader.dataset))
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=accuracy,
            num_examples=len(dataloader.dataset),
            metrics={}
        )
