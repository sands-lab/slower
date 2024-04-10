from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ControlCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OK: _ClassVar[ControlCode]
    DO_CLOSE_STREAM: _ClassVar[ControlCode]
    STREAM_CLOSED_OK: _ClassVar[ControlCode]
    ERROR_PROCESSING_STREAM: _ClassVar[ControlCode]
OK: ControlCode
DO_CLOSE_STREAM: ControlCode
STREAM_CLOSED_OK: ControlCode
ERROR_PROCESSING_STREAM: ControlCode

class BatchPredictionIns(_message.Message):
    __slots__ = ("embeddings", "control_code")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    embeddings: bytes
    control_code: ControlCode
    def __init__(self, embeddings: _Optional[bytes] = ..., control_code: _Optional[_Union[ControlCode, str]] = ...) -> None: ...

class BatchPredictionRes(_message.Message):
    __slots__ = ("predictions", "control_code")
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    predictions: bytes
    control_code: ControlCode
    def __init__(self, predictions: _Optional[bytes] = ..., control_code: _Optional[_Union[ControlCode, str]] = ...) -> None: ...

class GradientDescentDataBatchIns(_message.Message):
    __slots__ = ("embeddings", "labels", "control_code")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    embeddings: bytes
    labels: bytes
    control_code: ControlCode
    def __init__(self, embeddings: _Optional[bytes] = ..., labels: _Optional[bytes] = ..., control_code: _Optional[_Union[ControlCode, str]] = ...) -> None: ...

class GradientDescentDataBatchRes(_message.Message):
    __slots__ = ("gradient", "control_code")
    GRADIENT_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    gradient: bytes
    control_code: ControlCode
    def __init__(self, gradient: _Optional[bytes] = ..., control_code: _Optional[_Union[ControlCode, str]] = ...) -> None: ...

class UpdateServerSideModelRes(_message.Message):
    __slots__ = ("control_code",)
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    control_code: ControlCode
    def __init__(self, control_code: _Optional[_Union[ControlCode, str]] = ...) -> None: ...

class DataBatchForward(_message.Message):
    __slots__ = ("embeddings", "control_code")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    embeddings: bytes
    control_code: ControlCode
    def __init__(self, embeddings: _Optional[bytes] = ..., control_code: _Optional[_Union[ControlCode, str]] = ...) -> None: ...

class DataBatchBackward(_message.Message):
    __slots__ = ("gradient", "control_code")
    GRADIENT_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    gradient: bytes
    control_code: ControlCode
    def __init__(self, gradient: _Optional[bytes] = ..., control_code: _Optional[_Union[ControlCode, str]] = ...) -> None: ...
