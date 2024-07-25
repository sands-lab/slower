from typing import Optional, Tuple, Dict, Union

from flwr.server import ServerConfig
from flwr.server.client_manager import ClientManager, SimpleClientManager

from slower.server.strategy.base_strategy import SlStrategy
from slower.server.server import Server
from slower.server.server_model.manager.grpc_server_model_manager import (
    GrpcServerModelManager
)
try:
    from slower.server.server_model.manager.ray_private_server_model_manager import (
        RayPrivateServerModelManager
    )
    from slower.simulation.ray_transport.split_learning_actor_pool import (
        SplitLearningVirtualClientPool
    )
except ImportError as e:
    RayPrivateServerModelManager = None
    SplitLearningVirtualClientPool = None

#pylint: disable=too-many-arguments
def init_defaults(
    server: Optional[Server],
    config: Optional[ServerConfig],
    strategy: SlStrategy,
    client_manager: Optional[ClientManager],
    server_actor_resources: Optional[Dict[str, Union[float, int]]] = None,
    actor_pool = None
) -> Tuple[Server, ServerConfig]:
    """Create server instance if none was given."""

    assert bool(server_actor_resources is None) == (actor_pool is None)
    is_simulated_environment = actor_pool is not None

    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()

        if is_simulated_environment:
            assert RayPrivateServerModelManager is not None, \
                "You need to install flwr[simulation] to run the simulation environment"
            assert isinstance(actor_pool, SplitLearningVirtualClientPool)
            server_model_manager = RayPrivateServerModelManager(
                strategy.init_server_model_fn,
                actor_pool
            )
        else:
            server_model_manager = GrpcServerModelManager(
                init_server_model_fn=strategy.init_server_model_fn,
                server_model_resources={"num_cpus": 12},
                common_server_model=strategy.has_common_server_model()
            )

        server = Server(
            client_manager=client_manager,
            strategy=strategy,
            server_model_manager=server_model_manager
        )
    elif strategy is not None:
        print("Both server and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return server, config
