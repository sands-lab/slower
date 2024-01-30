from flwr.server import ServerConfig

from slower.server.app import start_server
from slower.server.strategy.plain_sl_strategy import PlainSlStrategy

import usage.cifar10.constants as constants
from usage.cifar10.raw.cifar_raw_server_segment import CifarRawServerSegment


def main():
    strategy = PlainSlStrategy(
        common_server=False,
        init_server_model_segment_fn=CifarRawServerSegment,
    )
    start_server(
        server_address=constants.SERVER_IP,
        strategy=strategy,
        config=ServerConfig(num_rounds=constants.N_EPOCHS)
    )


if __name__ == "__main__":
    main()
