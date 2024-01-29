import time
import random
from collections import OrderedDict

import torch
import numpy as np


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


class ExecutionTime:
    def __init__(self, l) -> None:
        self.l = l
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        self.l.append(execution_time)


def print_metrics(is_train, metrics_dict = {}, custom_values = {}):
    prefix = "Train" if is_train else "EVAL"

    for k, v in custom_values.items():
        print(f"{prefix} - {k}: {v}")
    for k, v in metrics_dict.items():
        print(f"{prefix} - {k}: {sum(v)}")


def init_complete_metrics_dict():
    return {
        "communication": [],
        "forward": [],
        "backward": [],
        "serialization": []
    }
