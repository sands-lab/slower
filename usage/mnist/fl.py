import flwr as fl

from usage.mnist.common import get_dataloader
from usage.mnist.models import ClientModel, ServerModel
from usage.common.helper import get_parameters, set_parameters
from usage.common.model import train, test_accuracy
import usage.mnist.constants as constants


class Client(fl.client.NumPyClient):
    def __init__(self, cid) -> None:

        super().__init__()
        self.cid = cid
        self.dataloader = get_dataloader()
        self.client_model = ClientModel()
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
        num_clients=constants.N_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=constants.N_EPOCHS),
        client_resources=constants.CLIENT_RESOURCES,
    )


if __name__ == "__main__":
    main()
