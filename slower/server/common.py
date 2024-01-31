from typing import Optional, Tuple, Dict, Union

from flwr.server import ServerConfig
from flwr.server.client_manager import ClientManager, SimpleClientManager

from slower.server.strategy.base_strategy import SlStrategy
from slower.server.server import Server
from slower.simulation.ray_transport.split_learning_actor_pool import SplitLearningVirtualClientPool
from slower.server.server_model_segment.manager.grpc_server_model_segment_manager import (
    GrpcServerModelSegmentManager
)
from slower.server.server_model_segment.manager.ray_common_server_model_segment_manager import (
    RayCommonServerModelSegmentManager
)
from slower.server.server_model_segment.manager.ray_private_server_model_segment_manager import (
    RayPrivateServerModelSegmentManager
)


#pylint: disable=too-many-arguments
def init_defaults(
    server: Optional[Server],
    config: Optional[ServerConfig],
    strategy: SlStrategy,
    client_manager: Optional[ClientManager],
    server_actor_resources: Optional[Dict[str, Union[float, int]]] = None,
    actor_pool: Optional[SplitLearningVirtualClientPool] = None
) -> Tuple[Server, ServerConfig]:
    """Create server instance if none was given."""

    assert bool(server_actor_resources is None) == (actor_pool is None)
    is_simulated_environment = actor_pool is not None

    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()

        if is_simulated_environment:
            if strategy.has_common_server_model_segment():
                sms_manager = RayCommonServerModelSegmentManager(
                    strategy.init_server_model_segment_fn,
                    server_model_segment_resources=server_actor_resources
                )
            else:
                sms_manager = RayPrivateServerModelSegmentManager(
                    strategy.init_server_model_segment_fn,
                    actor_pool
                )
        else:
            sms_manager = GrpcServerModelSegmentManager(
                init_server_model_segment_fn=strategy.init_server_model_segment_fn,
                server_model_segment_resources={"num_cpus": 12},
                common_server_model_segment=strategy.has_common_server_model_segment()
            )

        server = Server(
            client_manager=client_manager,
            strategy=strategy,
            server_model_segment_manager=sms_manager
        )
    elif strategy is not None:
        print("Both server and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return server, config
