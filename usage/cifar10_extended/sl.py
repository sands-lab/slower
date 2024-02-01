import os
from dotenv import load_dotenv

from flwr.server.app import ServerConfig

from slower.simulation.app import start_simulation
from slower.server.strategy import PlainSlStrategy


from usage.cifar10_extended.sl_client import CifarSlClient
from usage.cifar10_extended.sl_server_segment import CifarSlServerSegment
from usage.cifar10_extended.common import load_data_config
from usage.common.helper import seed
import usage.cifar10_extended.constants as constants


def main():

    load_dotenv("./usage/cifar10_extended/.env", verbose=True)
    config = load_data_config(os.getenv("PARTITION_FOLDER"))

    seed()

    client_fn = CifarSlClient
    fit_config_fn=lambda _: {"lr": constants.LR}
    strategy = PlainSlStrategy(
        common_server=constants.COMMON_SERVER,
        fraction_fit=constants.FRACTION_FIT,
        init_server_model_segment_fn=CifarSlServerSegment,
        config_server_segnent_fn=fit_config_fn,
        config_client_fit_fn=fit_config_fn
    )
    start_simulation(
        client_fn=client_fn,
        num_clients=config["num_clients"],
        strategy=strategy,
        client_resources=constants.CLIENT_RESOURCES,
        config=ServerConfig(num_rounds=constants.N_EPOCHS)
    )


if __name__ == "__main__":
    main()
