from typing import Optional
from transformers import BertModel

import torch
import torch.nn as nn

from usage.common.helper import seed
from usage.common.model import reset_parameters


def get_extended_attention_mask(
    attention_mask, input_shape, device: torch.device = None
):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
    extended_attention_mask.to(device)
    return extended_attention_mask

class ClientBert(nn.Module):
    def __init__(self, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        bert = BertModel.from_pretrained("bert-base-cased")
        self.embeddings = bert.embeddings
        self.encoder_layers = bert.encoder.layer[:n_layers]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)))
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long)
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=0
        )
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape)
        for layer_module in self.encoder_layers:
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask
            )
            hidden_states = layer_outputs[0]
        return hidden_states


class ServerBert(nn.Module):
    def __init__(self, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(0.1)
        self.encoder_layers = bert.encoder.layer[n_layers:]
        self.pooler = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Tanh()
        )
        self.classifier = nn.Linear(768, 5, bias=True)
        seed()
        reset_parameters(self.pooler)
        reset_parameters(self.classifier)

    def forward(
        self,
        hidden_states,
        attention_mask
    ):
        for layer_module in self.encoder_layers:
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
        hidden_states = hidden_states[:,0]
        hidden_states = self.pooler(hidden_states)
        hidden_states = self.dropout(hidden_states)
        preds = self.classifier(hidden_states)
        return preds
