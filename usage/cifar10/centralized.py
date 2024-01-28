import time

import torch

from models import get_client_model, get_server_model
from common import get_dataloader
from constants import N_CLIENT_LAYERS, N_EPOCHS


from usage.common.helper import seed
from usage.common.model import train, test_accuracy
from constants import CLIENT_RESOURCES

if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    start = time.time()
    seed()
    client_model = get_client_model(N_CLIENT_LAYERS)
    seed()
    server_model = get_server_model(N_CLIENT_LAYERS)
    print(client_model[0].weight[0,0,0])
    print(server_model[0].weight[0,0])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(client_model.parameters()) + list(server_model.parameters()), lr=0.05)
    dataloader = get_dataloader()
    for _ in range(N_EPOCHS):
        train(client_model, server_model, dataloader)
        acc = test_accuracy(client_model, server_model, dataloader)
        print(acc)

    print(f"Total time: {time.time() - start}")
