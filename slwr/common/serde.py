from typing import Dict, List, Union

from slwr.proto import server_model_pb2
from slwr.proto.server_model_pb2 import ControlCode as GrpcControlCode
from slwr.common.typing import ControlCode


def control_code_to_proto(in_code: ControlCode) -> GrpcControlCode:
    """Serialize `Status` to ProtoBuf."""
    out_code = GrpcControlCode.OK
    if in_code == ControlCode.DO_CLOSE_STREAM:
        out_code = GrpcControlCode.DO_CLOSE_STREAM
    elif in_code == ControlCode.ERROR_PROCESSING_STREAM:
        out_code = GrpcControlCode.ERROR_PROCESSING_STREAM
    elif in_code == ControlCode.STREAM_CLOSED_OK:
        out_code = GrpcControlCode.STREAM_CLOSED_OK
    elif in_code == ControlCode.INIT_STREAM:
        out_code = GrpcControlCode.INIT_STREAM
    return out_code


def control_code_from_proto(in_code: GrpcControlCode) -> ControlCode:
    """Deserialize `Status` from ProtoBuf."""
    out_code = ControlCode.OK
    if in_code == GrpcControlCode.STREAM_CLOSED_OK:
        out_code = ControlCode.STREAM_CLOSED_OK
    elif in_code == GrpcControlCode.DO_CLOSE_STREAM:
        out_code = ControlCode.DO_CLOSE_STREAM
    elif in_code == GrpcControlCode.ERROR_PROCESSING_STREAM:
        out_code = ControlCode.ERROR_PROCESSING_STREAM
    elif in_code == GrpcControlCode.INIT_STREAM:
        out_code = ControlCode.INIT_STREAM
    return out_code


def from_grpc_format(
    data: Dict[str, server_model_pb2.ByteTensor]
) -> Dict[str, Union[bytes, List[bytes]]]:
    """Convert a batch of data in protobuf format to a native python dictionary

    Parameters
    ----------
    data : Dict[str, server_model_pb2.ByteTensor]
        Data to be deserialized

    Returns
    -------
    Dict[str, Union[bytes, List[bytes]]]
        Deserialized data
    """
    out = {}
    for key, value in data.items():
        field = value.WhichOneof("data")
        if field == "tensors":
            out[key] = list(value.tensors.tensors)
        else:
            out[key] = value.single_tensor
    return out


def to_grpc_format(
    data: Dict[str, Union[bytes, List[bytes]]]
) -> Dict[str, server_model_pb2.ByteTensor]:
    """Serialize a batch of data to protobuf format

    Parameters
    ----------
    data : Dict[str, Union[bytes, List[bytes]]]
        Data to be serialized

    Returns
    -------
    Dict[str, server_model_pb2.ByteTensor]
        Serialized data
    """
    out = {}
    for key, value in data.items():
        if isinstance(value, bytes):
            out[key] = server_model_pb2.ByteTensor(single_tensor=value)
        else:
            out[key] = server_model_pb2.ByteTensor(
                tensors=server_model_pb2.TensorList(tensors=value)
            )
    return out
