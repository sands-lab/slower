
from flwr.server.app import ServerConfig

from slower.simulation.app import start_simulation
from slower.server.strategy import PlainSlStrategy

from usage.mnist.mnist_client import MnistClient
from usage.mnist.server_segment import SimpleServerModelSegment


def main():
    client_fn = MnistClient
    strategy = PlainSlStrategy(
        common_server=False,
        init_server_model_segment_fn=SimpleServerModelSegment,
    )
    start_simulation(
        client_fn=client_fn,
        num_clients=1,
        strategy=strategy,
        client_resources={"num_cpus":2, "num_gpus": 0.},
        config=ServerConfig(num_rounds=4)
    )


if __name__ == "__main__":
    main()
