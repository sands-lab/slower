from flwr.client import Client, NumPyClient as FlwrNumPyClient
from flwr.client.numpy_client import _wrap_numpy_client


class NumPyClient(FlwrNumPyClient):
    """Abstract base class for Flower clients using NumPy."""

    def to_client(self) -> Client:
        """Convert object to Client type and return it."""
        client = _wrap_numpy_client(client=self)
        client.numpy_client.server_model_proxy = None
        client.set_server_model_proxy = \
            lambda smp: setattr(client.numpy_client, 'server_model_proxy', smp)
        return client
