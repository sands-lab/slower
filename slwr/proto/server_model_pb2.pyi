from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ControlCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OK: _ClassVar[ControlCode]
    DO_CLOSE_STREAM: _ClassVar[ControlCode]
    STREAM_CLOSED_OK: _ClassVar[ControlCode]
    ERROR_PROCESSING_STREAM: _ClassVar[ControlCode]
    INIT_STREAM: _ClassVar[ControlCode]
OK: ControlCode
DO_CLOSE_STREAM: ControlCode
STREAM_CLOSED_OK: ControlCode
ERROR_PROCESSING_STREAM: ControlCode
INIT_STREAM: ControlCode

class BatchData(_message.Message):
    __slots__ = ("method", "data", "control_code", "cid")
    class DataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ByteTensor
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ByteTensor, _Mapping]] = ...) -> None: ...
    METHOD_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    CID_FIELD_NUMBER: _ClassVar[int]
    method: str
    data: _containers.MessageMap[str, ByteTensor]
    control_code: ControlCode
    cid: str
    def __init__(self, method: _Optional[str] = ..., data: _Optional[_Mapping[str, ByteTensor]] = ..., control_code: _Optional[_Union[ControlCode, str]] = ..., cid: _Optional[str] = ...) -> None: ...

class ByteTensor(_message.Message):
    __slots__ = ("single_tensor", "tensors")
    SINGLE_TENSOR_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    single_tensor: bytes
    tensors: TensorList
    def __init__(self, single_tensor: _Optional[bytes] = ..., tensors: _Optional[_Union[TensorList, _Mapping]] = ...) -> None: ...

class TensorList(_message.Message):
    __slots__ = ("tensors",)
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    tensors: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, tensors: _Optional[_Iterable[bytes]] = ...) -> None: ...
