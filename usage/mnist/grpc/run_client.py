from slower.client.app import start_client
from usage.mnist.numpy.mnist_numpy_client import MnistNumpyClient
import usage.mnist.constants as constants


def main():
    start_client(server_address=constants.SERVER_IP, client_fn= MnistNumpyClient)


if __name__ == "__main__":
    main()
