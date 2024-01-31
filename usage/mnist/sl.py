
from flwr.server.app import ServerConfig

from slower.simulation.app import start_simulation
from slower.server.strategy import PlainSlStrategy

from usage.mnist.numpy.mnist_numpy_client import MnistNumpyClient
from usage.mnist.numpy.mnist_numpy_server_segment import MnistNumpyServerSegment

from usage.mnist.raw.mnist_raw_client import MnistRawClient
from usage.mnist.raw.mnist_raw_server_segment import MnistRawServerSegment

import usage.mnist.constants as constants


def main():
    if constants.USE_NUMPY_CLIENTS:
        client_fn = MnistNumpyClient
        strategy = PlainSlStrategy(
            common_server=constants.COMMON_SERVER,
            init_server_model_segment_fn=MnistNumpyServerSegment,
        )
    else:
        client_fn = MnistRawClient
        strategy = PlainSlStrategy(
            common_server=constants.COMMON_SERVER,
            init_server_model_segment_fn=MnistRawServerSegment,
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
