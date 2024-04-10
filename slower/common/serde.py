from slower.proto.server_segment_pb2 import ControlCode as GrpcControlCode
from slower.common.typing import ControlCode

def control_code_to_proto(in_code: ControlCode) -> GrpcControlCode:
    """Serialize `Status` to ProtoBuf."""
    out_code = GrpcControlCode.OK
    if in_code == ControlCode.DO_CLOSE_STREAM:
        out_code = GrpcControlCode.DO_CLOSE_STREAM
    elif in_code == ControlCode.ERROR_PROCESSING_STREAM:
        out_code = GrpcControlCode.ERROR_PROCESSING_STREAM
    elif in_code == ControlCode.STREAM_CLOSED_OK:
        out_code = GrpcControlCode.STREAM_CLOSED_OK
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
    return out_code
