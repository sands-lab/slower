from slower.client.app import start_client

from usage.cifar10.raw.cifar_raw_client import CifarRawClient
import usage.cifar10.constants as constants


def main():
    start_client(server_address=constants.SERVER_IP, client_fn=CifarRawClient)


if __name__ == "__main__":
    main()
