from flwr.server import ServerConfig

from slower.server.app import start_server
from slower.server.strategy.plain_sl_strategy import PlainSlStrategy

import usage.yelp_review.constants as constants
from usage.yelp_review.raw.yelp_raw_server_segment import YelpRawServerSegment


def main():
    strategy = PlainSlStrategy(
        common_server=False,
        init_server_model_segment_fn=YelpRawServerSegment,
    )
    start_server(
        server_address=constants.SERVER_IP,
        strategy=strategy,
        config=ServerConfig(num_rounds=constants.N_EPOCHS)
    )


if __name__ == "__main__":
    main()
