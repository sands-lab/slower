import time

import torch

from models import ServerModel, ClientModel
from common import get_dataloader

from usage.common.helper import seed
from usage.common.model import train, test_accuracy


if __name__ == "__main__":
    start = time.time()
    seed()
    client_model = ClientModel()
    seed()
    server_model = ServerModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(client_model.parameters()) + list(server_model.parameters()), lr=0.05)
    dataloader = get_dataloader()
    for _ in range(4):
        train(client_model, server_model, dataloader)
        acc = test_accuracy(client_model, server_model, dataloader)
        print(acc)

    print(f"Total time: {time.time() - start}")
