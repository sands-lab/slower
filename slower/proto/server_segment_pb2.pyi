from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class BatchPredictionIns(_message.Message):
    __slots__ = ("embeddings", "cid")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    CID_FIELD_NUMBER: _ClassVar[int]
    embeddings: bytes
    cid: str
    def __init__(self, embeddings: _Optional[bytes] = ..., cid: _Optional[str] = ...) -> None: ...

class BatchPredictionRes(_message.Message):
    __slots__ = ("predictions",)
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    predictions: bytes
    def __init__(self, predictions: _Optional[bytes] = ...) -> None: ...

class GradientDescentDataBatchIns(_message.Message):
    __slots__ = ("embeddings", "labels", "cid")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    CID_FIELD_NUMBER: _ClassVar[int]
    embeddings: bytes
    labels: bytes
    cid: str
    def __init__(self, embeddings: _Optional[bytes] = ..., labels: _Optional[bytes] = ..., cid: _Optional[str] = ...) -> None: ...

class GradientDescentDataBatchRes(_message.Message):
    __slots__ = ("gradient",)
    GRADIENT_FIELD_NUMBER: _ClassVar[int]
    gradient: bytes
    def __init__(self, gradient: _Optional[bytes] = ...) -> None: ...
