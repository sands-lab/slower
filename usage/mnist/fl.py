import flwr as fl

from common import get_dataloader
from models import ClientModel, ServerModel

from usage.common.helper import get_parameters, set_parameters, seed
from usage.common.model import train, test_accuracy


class Client(fl.client.NumPyClient):
    def __init__(self, cid) -> None:

        super().__init__()
        self.dataloader = get_dataloader()
        seed()
        self.client_model = ClientModel()
        seed()
        self.server_model = ServerModel()

    def get_parameters(self, config):
        return get_parameters(self.client_model) + get_parameters(self.server_model)

    def set_parameters(self, parameters):
        n_client = len(get_parameters(self.client_model))
        set_parameters(self.client_model, parameters[:n_client])
        set_parameters(self.server_model, parameters[n_client:])

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = test_accuracy(self.client_model, self.server_model, self.dataloader)
        return float(acc), len(self.dataloader.dataset), {}

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.client_model, self.server_model, self.dataloader)
        return self.get_parameters(config={}), len(self.dataloader.dataset), {}


def main():
    fl.simulation.start_simulation(
        client_fn=Client,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=4),
        client_resources={"num_cpus": 2, "num_gpus": 0.},
    )


if __name__ == "__main__":
    main()
