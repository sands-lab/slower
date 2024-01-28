from typing import Callable

from slower.client.client import Client


ClientFn = Callable[[str], Client]
