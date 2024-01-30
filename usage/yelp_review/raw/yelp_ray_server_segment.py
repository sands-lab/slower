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
    bytes_to_torch,
    bytes_to_torch_list,
    torch_list_to_bytes
)
from slower.server.server_model_segment.server_model_segment import ServerModelSegment

from usage.yelp_review.models import ServerBert
from usage.yelp_review.constants import N_CLIENT_LAYERS
from usage.yelp_review.compression import compress_embeddings, uncompress_embeddings, get_extended_attention_mask
from usage.common.helper import seed, set_parameters, get_parameters


class YelpRawServerSegment(ServerModelSegment):

    def __init__(self) -> None:
        seed()
        self.model = ServerBert(N_CLIENT_LAYERS)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.model.eval()

    def serve_prediction_request(self, batch_data: BatchPredictionIns) -> BatchPredictionRes:
        embeddings = bytes_to_torch_list(batch_data.embeddings)
        embeddings, lens = uncompress_embeddings(embeddings)

        with torch.no_grad():
           preds = self.model(
                hidden_states = embeddings,
                attention_mask = get_extended_attention_mask(embeddings.shape[0], embeddings.shape[1], lens)
            )
        preds = torch.argmax(preds, axis=1)
        preds = torch_to_bytes(preds)
        return BatchPredictionRes(preds)

    def serve_gradient_update_request(
        self,
        batch_data: GradientDescentDataBatchIns
    ) -> GradientDescentDataBatchRes:
        embeddings = bytes_to_torch_list(batch_data.embeddings)
        labels = bytes_to_torch(batch_data.labels, False)
        embeddings, lens = uncompress_embeddings(embeddings)
        embeddings.requires_grad_(True)
        preds = self.model(
            hidden_states = embeddings,
            attention_mask = get_extended_attention_mask(embeddings.shape[0], embeddings.shape[1], lens)
        )
        loss = self.criterion(preds, labels)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        error = embeddings.grad
        error = compress_embeddings(error, lens)

        return GradientDescentDataBatchRes(gradient=torch_list_to_bytes(error))

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
