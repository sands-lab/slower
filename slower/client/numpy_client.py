# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client app."""


from typing import Callable, Dict

from flwr.client import Client, NumPyClient as FlwrNumPyClient
from flwr.client.numpy_client import (
    _fit,
    _evaluate,
    _constructor,
    _get_parameters,
    _get_properties,
    has_fit,
    has_evaluate,
    has_get_parameters,
    has_get_properties,
)
from flwr.client.workload_state import WorkloadState

from slower.server.server_model.proxy.server_model_proxy import ServerModelProxy


class NumPyClient(FlwrNumPyClient):
    """Abstract base class for Flower clients using NumPy."""

    state: WorkloadState

    def __init__(self) -> None:
        self.server_model_proxy = None

    def to_client(self) -> Client:
        """Convert to object to Client type and return it."""
        return _wrap_numpy_client(client=self)

    def set_server_model_proxy(self, server_model_proxy: ServerModelProxy):
        self.server_model_proxy = server_model_proxy


def _set_server_model_proxy(self: Client, server_model_proxy: ServerModelProxy):
    self.numpy_client.set_server_model_proxy(server_model_proxy)


def _wrap_numpy_client(client: NumPyClient) -> Client:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
        "cid": client.cid,
        "set_server_model_proxy": _set_server_model_proxy,
    }

    # Add wrapper type methods (if overridden)
    if has_get_parameters(client=client):
        member_dict["get_parameters"] = _get_parameters

    if has_fit(client=client):
        member_dict["fit"] = _fit

    if has_evaluate(client=client):
        member_dict["evaluate"] = _evaluate

    if has_get_properties(client=client):
        member_dict["get_properties"] = _get_properties

    # Create wrapper class
    wrapper_class = type("NumPyClientWrapper", (Client,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_client=client)  # type: ignore
