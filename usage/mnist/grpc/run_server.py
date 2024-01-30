from flwr.server import ServerConfig

from slower.server.app import start_server
from slower.server.strategy.plain_sl_strategy import PlainSlStrategy

import usage.mnist.constants as constants
from usage.mnist.numpy.mnist_numpy_server_segment import MnistNumpyServerSegment


def main():
    strategy = PlainSlStrategy(
        common_server=False,
        init_server_model_segment_fn=MnistNumpyServerSegment,
    )
    start_server(
        server_address=constants.SERVER_IP,
        strategy=strategy,
        config=ServerConfig(num_rounds=constants.N_EPOCHS)
    )


if __name__ == "__main__":
    main()
