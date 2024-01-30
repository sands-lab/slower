import flwr as fl

import torch
from usage.yelp_review.common import get_dataloader
from usage.yelp_review.models import ClientBert, ServerBert
import usage.yelp_review.constants as constants
from usage.common.helper import set_parameters, get_parameters, seed


def _train(client_model, server_model, dataloader):
    optimizer = torch.optim.SGD(list(client_model.parameters()) + list(server_model.parameters()), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    client_model.eval()
    server_model.eval()
    for batch in dataloader:
        embeddings = client_model(**{k: v for k, v in batch.items() if k != "labels"})
        preds = server_model(hidden_states=embeddings, attention_mask=batch["attention_mask"].unsqueeze(1).unsqueeze(1))
        loss = criterion(preds, batch["labels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _evaluate(client_model, server_model, dataloader):
    corrects = 0
    for batch in dataloader:
        with torch.no_grad():
            embeddings = client_model(**{k: v for k, v in batch.items() if k != "labels"})
            preds = server_model(hidden_states=embeddings, attention_mask=batch["attention_mask"].unsqueeze(1).unsqueeze(1))
        preds = torch.argmax(preds, dim=1)
        corrects += (preds == batch["labels"]).int().sum()
    return corrects / len(dataloader.dataset)


class Client(fl.client.NumPyClient):
    def __init__(self, cid) -> None:

        super().__init__()
        self.dataloader = get_dataloader()
        self.client_model = ClientBert(constants.N_CLIENT_LAYERS)
        self.server_model = ServerBert(constants.N_CLIENT_LAYERS)
        self.client_model.eval()
        self.server_model.eval()

    def get_parameters(self, config):
        return get_parameters(self.client_model) + get_parameters(self.server_model)

    def set_parameters(self, parameters):
        n_client = len(get_parameters(self.client_model))
        set_parameters(self.client_model, parameters[:n_client])
        set_parameters(self.server_model, parameters[n_client:])

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = _evaluate(self.client_model, self.server_model, self.dataloader)
        return float(acc), len(self.dataloader.dataset), {}

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        _train(self.client_model, self.server_model, self.dataloader)
        return self.get_parameters(config={}), len(self.dataloader.dataset), {}


def main():
    fl.simulation.start_simulation(
        client_fn=Client,
        num_clients=constants.N_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=constants.N_EPOCHS),
        client_resources=constants.CLIENT_RESOURCES
    )


if __name__ == "__main__":
    main()
