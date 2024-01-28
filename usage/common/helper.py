import torch
import random
import numpy as np
from collections import OrderedDict


def seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
