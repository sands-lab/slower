import threading
from logging import INFO
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from flwr.common.logger import log
from flwr.simulation.ray_transport.ray_actor import (
    VirtualClientEngineActor,
    VirtualClientEngineActorPool,
    pool_size_from_resources,
)
from ray import ObjectRef
from ray.util import ActorPool


class SplitLearningVirtualClientPool(VirtualClientEngineActorPool):

    #pylint: disable=non-parent-init-called
    #pylint: disable=super-init-not-called
    def __init__(
        self,
        create_actor_fn: Callable[[], Type[VirtualClientEngineActor]],
        common_server_model_segment: bool,
        client_resources: Dict[str, Union[int, float]],
        actor_list: Optional[List[Type[VirtualClientEngineActor]]] = None,
    ):
        self.client_resources = client_resources
        self.create_actor_fn = create_actor_fn

        if actor_list is None:
            # Figure out how many actors can be created given the cluster resources
            # and the resources the user indicates each VirtualClient will need
            num_actors = pool_size_from_resources(client_resources)
            if common_server_model_segment:

                # TODO common server trainer should have more resources than individual clients!!!
                num_actors -= 1
                log(INFO, "Decreasing number of actors in pool to %d", num_actors)
            actors = [create_actor_fn() for _ in range(num_actors)]
        else:
            # When __reduce__ is executed, we don't want to created
            # a new list of actors again.
            actors = actor_list

        ActorPool.__init__(self, actors)  # TODO: is doing this ok?

        # A dict that maps cid to another dict containing: a reference to the remote job
        # and its status (i.e. whether it is ready or not)
        self._cid_to_future: Dict[
            str, Dict[str, Union[bool, Optional[ObjectRef[Any]]]]
        ] = {}
        self.actor_to_remove: Set[str] = set()  # a set
        self.num_actors = len(actors)

        self.lock = threading.RLock()

        self.common_server_model_segment = common_server_model_segment
        self.reset_object_reference_mapping()

    def reset_object_reference_mapping(self):
        """Remove all existing references to the ray memory, so that the objects can get evicted"""
        with self.lock:
            self._object_reference_mapping= {}

    def add_object_reference_mapping(self, cid: str, object_ref: ObjectRef):
        """When requested, store a reference to an object in the shared ray memory, so that the
        object does not get evicted"""
        with self.lock:
            self._object_reference_mapping[cid] = object_ref

    def __reduce__(self):  # type: ignore
        """Make this class serializable (needed due to lock)."""
        return SplitLearningVirtualClientPool, (
            self.create_actor_fn,
            self.common_server_model_segment,
            self.client_resources,
            self._idle_actors,  # Pass existing actors to avoid killing/re-creating
        )
