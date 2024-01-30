import torch.nn as nn

from usage.common.model import reset_parameters
from usage.common.helper import seed


_layers = [
    nn.Conv2d(3, 32, 3, 2, 1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 2, 1),
    nn.ReLU(),
    nn.Conv2d(64, 96, 3, 2, 1),
    nn.ReLU(),
    nn.Conv2d(96, 128, 3, 2, 1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(128 * 4, 10)
]


def get_client_model(n_trainable_client_layers):
    assert n_trainable_client_layers < 4, "At most 4 layers can be trained on the client"
    model = nn.Sequential(*_layers[:n_trainable_client_layers])
    seed()
    reset_parameters(model)
    return model


def get_server_model(n_trainable_client_layers):
    assert n_trainable_client_layers < 4, "At most 4 layers can be trained on the client"
    model =  nn.Sequential(*_layers[n_trainable_client_layers:])
    seed()
    reset_parameters(model)
    return model
