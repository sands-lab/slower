
from flwr.server.app import ServerConfig

from slower.simulation.app import start_simulation
from slower.server.strategy import PlainSlStrategy


from usage.yelp_review.raw.yelp_raw_client import YelpRawClient
from usage.yelp_review.raw.yelp_raw_server_segment import YelpRawServerSegment
import usage.yelp_review.constants as constants


def main():
    client_fn = YelpRawClient
    strategy = PlainSlStrategy(
        common_server=False,
        init_server_model_segment_fn=YelpRawServerSegment,
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
