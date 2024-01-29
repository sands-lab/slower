import ray
from slower.server.app import start_server
from usage.mnist.server_segment import SimpleServerModelSegment

from slower.server.strategy.plain_sl_strategy import PlainSlStrategy
from flwr.server import ServerConfig


address = "10.109.66.187:8080"
strategy = PlainSlStrategy(
    common_server=False,
    init_server_model_segment_fn=SimpleServerModelSegment,
)
ray.init()
process_server = start_server(server_address = address, strategy = strategy, config=ServerConfig(num_rounds=4))
ray.shutdown()
