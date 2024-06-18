from enum import Enum
from typing import Dict, Union, List, Optional
from abc import ABC, abstractmethod

import numpy as np

from slower.common import (
    ControlCode,
    BatchData,
)
from slower.common.constants import RETURN_TENSOR_TYPE_KEY
from slower.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict


NumpyBatchData = Dict[str, Union[np.ndarray, List[np.ndarray]]]
FORBIDDEN_SERVER_MODEL_METHODS = {"get_parameters", "configure_fit", "configure_evaluate",
                                  "to_server_model", "get_synchronization_result"}


class _RequestFormat(Enum):
    RAW = 1
    BYTES = 2
    NUMPY = 3

    def __eq__(self, other):
        return self.value == other.value


class ServerModelProxy(ABC):

    @abstractmethod
    def _blocking_request(
        self,
        method: str,
        batch_data: BatchData,
        timeout: Optional[float]
    ) -> BatchData:
        """Issue a blocking request to the server model

        Parameters
        ----------
        method : str
            Name of the server method to be invoked
        batch_data : BatchData
            Request data
        timeout : Optional[float]
            Maximum time allows for the request to complete

        Returns
        -------
        BatchData
            Data returned by the server model

        Raises
        ------
        AttributeError
            If the server model does not have the requested method
        Exception
            If any other exception occurs during the request
        """

    @abstractmethod
    def _streaming_request(self, method: str, batch_data: BatchData) -> None:
        """Issue a non blocking request to the server model

        Parameters
        ----------
        method : str
            Name of the server method to be invoked
        batch_data : BatchData
            Request data
        """

    @abstractmethod
    def _initialize_stream(self):
        """Initialize the stream of requests to be submitted to the server model
        """

    @abstractmethod
    def _is_stream_initialized(self):
        """Returns True if the stream in the streaming API is initialized else False
        """

    @abstractmethod
    def close_stream(self) -> BatchData:
        """Wait until the server finishes processing when streaming data to it

        Returns
        -------
        BatchData
            Data returned by the server
        """

    def numpy_close_stream(self) -> np.ndarray:
        res = self.close_stream()
        return bytes_to_ndarray_dict(res.data)

    @abstractmethod
    def get_pending_batches_count(self) -> int:
        """In the streaming API, returns the number of batches that are unprocessed (i.e., how much is the client ahead of the server)

        Returns
        -------
        int
            number of pending batches
        """

    def __check_dtype(self, d, base_type):
        for value in d.values():
            if isinstance(value, base_type):
                continue
            elif isinstance(value, list) and all(isinstance(item, base_type) for item in value):
                continue
            else:
                return False
        return True

    def __parse_request_args(self, *args, **kwargs):
        if len(args) == 1:
            request = args[0]
            assert isinstance(request, BatchData)
            request_format = _RequestFormat.RAW

        else:
            assert self.__check_dtype(kwargs, np.ndarray)
            data = ndarray_dict_to_bytes(kwargs)
            request_format = _RequestFormat.NUMPY

            request = BatchData(data=data, control_code=ControlCode.OK)

        return request, request_format

    def _parse_response_args(self, batch_data: BatchData, request_format: _RequestFormat):
        if request_format == _RequestFormat.NUMPY:

            res = bytes_to_ndarray_dict(batch_data.data)
            tp = res.pop(RETURN_TENSOR_TYPE_KEY)
            return res[""] if tp == "np" else res
        elif request_format == _RequestFormat.RAW:
            return batch_data

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if name in FORBIDDEN_SERVER_MODEL_METHODS:
            raise Exception(f"Client should not invoke the {name} method")

        def _request(*args, blocking=True, timeout=None, **kwargs):
            batch_data, request_format = self.__parse_request_args(*args, **kwargs)
            if blocking:
                res = self._blocking_request(method=name, batch_data=batch_data, timeout=timeout)
                res = self._parse_response_args(res, request_format)
                return res
            else:
                if not self._is_stream_initialized():
                    self._initialize_stream()
                self._streaming_request(method=name, batch_data=batch_data)
        return _request
