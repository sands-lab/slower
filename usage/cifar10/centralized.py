import time

from usage.cifar10.models import get_client_model, get_server_model
from usage.cifar10.common import get_dataloader
from usage.common.model import train, test_accuracy
import usage.cifar10.constants as constants



def main():
    start = time.time()
    client_model = get_client_model(constants.N_CLIENT_LAYERS)
    server_model = get_server_model(constants.N_CLIENT_LAYERS)

    dataloader = get_dataloader()
    for _ in range(constants.N_EPOCHS):
        train(client_model, server_model, dataloader)
        acc = test_accuracy(client_model, server_model, dataloader)
        print(acc)

    print(f"Total time: {time.time() - start}")

if __name__ == "__main__":
    main()
