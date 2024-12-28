from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.client.typing import ClientFn
from flwr.common import Context, Message
from flwr.common.constant import MessageType, MessageTypeLegacy
from flwr.common.recordset_compat import (
    evaluateres_to_recordset,
    fitres_to_recordset,
    getparametersres_to_recordset,
    getpropertiesres_to_recordset,
    recordset_to_evaluateins,
    recordset_to_fitins,
    recordset_to_getparametersins,
    recordset_to_getpropertiesins,
)

from slwr.common.constants import CLIENT_ID_CONFIG_KEY
from slwr.proto.server_model_pb2_grpc import ServerModelStub
from slwr.server.server_model.proxy.grpc_server_model_proxy import GrpcServerModelProxy


def handle_legacy_message_from_msgtype(
    client_fn: ClientFn, message: Message, context: Context, server_model_stub: ServerModelStub
) -> Message:
    """Handle legacy message in the inner most mod."""
    cid = str(message.metadata.partition_id)
    assert len(message.content.configs_records) == 1
    config_record = next(iter(message.content.configs_records.values()))

    server_cid = ""
    if CLIENT_ID_CONFIG_KEY in config_record:
        server_cid = config_record.pop(CLIENT_ID_CONFIG_KEY)

    client = client_fn(cid)
    client = client.to_client()

    server_model_proxy = GrpcServerModelProxy(stub=server_model_stub, cid=server_cid)
    client.set_server_model_proxy(
        server_model_proxy
    )
    client.set_context(context)

    message_type = message.metadata.message_type

    # Handle GetPropertiesIns
    if message_type == MessageTypeLegacy.GET_PROPERTIES:
        get_properties_res = maybe_call_get_properties(
            client=client,
            get_properties_ins=recordset_to_getpropertiesins(message.content),
        )
        out_recordset = getpropertiesres_to_recordset(get_properties_res)
    # Handle GetParametersIns
    elif message_type == MessageTypeLegacy.GET_PARAMETERS:
        get_parameters_res = maybe_call_get_parameters(
            client=client,
            get_parameters_ins=recordset_to_getparametersins(message.content),
        )
        out_recordset = getparametersres_to_recordset(
            get_parameters_res, keep_input=False
        )
    # Handle FitIns
    elif message_type == MessageType.TRAIN:
        fit_res = maybe_call_fit(
            client=client,
            fit_ins=recordset_to_fitins(message.content, keep_input=True),
        )
        out_recordset = fitres_to_recordset(fit_res, keep_input=False)
    # Handle EvaluateIns
    elif message_type == MessageType.EVALUATE:
        evaluate_res = maybe_call_evaluate(
            client=client,
            evaluate_ins=recordset_to_evaluateins(message.content, keep_input=True),
        )
        out_recordset = evaluateres_to_recordset(evaluate_res)
    else:
        raise ValueError(f"Invalid message type: {message_type}")

    server_model_proxy.close_stream()

    # Return Message
    return message.create_reply(out_recordset)
