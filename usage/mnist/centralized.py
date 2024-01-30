import time

from models import ServerModel, ClientModel
from common import get_dataloader

from usage.common.model import train, test_accuracy
import usage.mnist.constants as constants


def main():
    start = time.time()
    client_model = ClientModel()
    server_model = ServerModel()

    dataloader = get_dataloader()
    for _ in range(constants.N_EPOCHS):
        train(client_model, server_model, dataloader)
        acc = test_accuracy(client_model, server_model, dataloader)
        print(acc)

    print(f"Total time: {time.time() - start}")


if __name__ == "__main__":
    main()
