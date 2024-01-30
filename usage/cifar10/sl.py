
from flwr.server.app import ServerConfig

from slower.simulation.app import start_simulation
from slower.server.strategy import PlainSlStrategy


from usage.cifar10.raw.cifar_raw_client import CifarRawClient
from usage.cifar10.raw.cifar_raw_server_segment import CifarRawServerSegment
import usage.cifar10.constants as constants


def main():
    client_fn = CifarRawClient
    strategy = PlainSlStrategy(
        common_server=constants.COMMON_SERVER,
        init_server_model_segment_fn=CifarRawServerSegment,
    )
    start_simulation(
        client_fn=client_fn,
        num_clients=constants.N_CLIENTS,
        strategy=strategy,
        client_resources=constants.CLIENT_RESOURCES,
        config=ServerConfig(num_rounds=constants.N_EPOCHS)
    )


if __name__ == "__main__":
    main()
