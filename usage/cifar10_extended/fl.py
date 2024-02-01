import os

import torch
import flwr as fl
from dotenv import load_dotenv

from usage.cifar10_extended.common import get_dataloader, load_data_config
from usage.cifar10_extended.models import (
    get_server_model,
    get_client_model,
    train,
    test_accuracy
)
import usage.cifar10_extended.constants as constants

from usage.common.helper import set_parameters, get_parameters, seed


class Client(fl.client.NumPyClient):
    def __init__(self, cid) -> None:

        super().__init__()
        self.cid = cid
        self.client_model = get_client_model(constants.N_CLIENT_LAYERS)
        self.server_model = get_server_model(constants.N_CLIENT_LAYERS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_parameters(self.client_model) + get_parameters(self.server_model)

    def set_parameters(self, parameters):
        n_client = len(get_parameters(self.client_model))
        set_parameters(self.client_model, parameters[:n_client])
        set_parameters(self.server_model, parameters[n_client:])

    def evaluate(self, parameters, config):
        testloader = get_dataloader(False, self.cid)
        self.set_parameters(parameters)
        acc = test_accuracy(self.client_model, self.server_model, testloader, self.device)
        return float(acc), len(testloader.dataset), {}

    def fit(self, parameters, config):
        print(f"Training client {self.cid}")
        trainloader = get_dataloader(True, self.cid)
        self.set_parameters(parameters)
        train(self.client_model, self.server_model, trainloader, config["lr"], self.device)
        return self.get_parameters(config={}), len(trainloader.dataset), {}


def main():
    load_dotenv("./usage/cifar10_extended/.env", verbose=True)
    config = load_data_config(os.getenv("PARTITION_FOLDER"))

    seed()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=constants.FRACTION_FIT,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        on_fit_config_fn=lambda _: {"lr": constants.LR}
    )

    fl.simulation.start_simulation(
        client_fn=Client,
        strategy=strategy,
        num_clients=config["num_clients"],
        config=fl.server.ServerConfig(num_rounds=constants.N_EPOCHS),
        client_resources=constants.CLIENT_RESOURCES
    )


if __name__ == "__main__":
    main()
