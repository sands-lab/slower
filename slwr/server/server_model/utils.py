from typing import Union
from functools import partial

import numpy as np
try:
    import torch
except ImportError:
    torch = None


def split_numpy(data, split_sizes: Union[int, np.ndarray]):
    _size = split_sizes if isinstance(split_sizes, int) else split_sizes[:-1]
    if isinstance(data, np.ndarray):
        return np.split(data, _size, axis=0)
    if isinstance(data, list):
        tmp = [np.split(d, _size, axis=0) for d in data]
        return [list(t) for t in zip(*tmp)]
    else:
        raise ValueError(f"Unsupported type {type(data)}")

def _torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def pytorch_format(func):
    def wrapper(self, **kwargs):
        assert torch is not None
        assert len(kwargs) > 0, "No arguments provided to the function"
        torch_kwargs = {}
        sizes = None
        for kw, values in kwargs.items():
            tmp = [torch.from_numpy(v) for v in values]

            if sizes is None:
                sizes = [v.shape[0] for v in values]
            else:
                assert sizes == [v.shape[0] for v in values]

            torch_kwargs[kw] = torch.cat(tmp, dim=0)

        res = func(self, **torch_kwargs)

        cum_sizes = np.cumsum(sizes)
        split_fn = partial(split_numpy, split_sizes=cum_sizes)

        if res is None or len(res) == 0:
            # assume that there is a streaming request
            # TODO: how do enforce that methods invoked with streaming requests return None?
            return None

        if isinstance(res, torch.Tensor):
            return split_fn(_torch_to_numpy(res))
        elif isinstance(res, list):
            return split_fn([_torch_to_numpy(v) for v in res])

        assert isinstance(res, dict)

        out_list = [{} for _ in range(len(sizes))]
        for key, value in res.items():
            if isinstance(value, torch.Tensor):
                split_value = split_fn(_torch_to_numpy(value))
            elif isinstance(value, list):
                split_value = split_fn([_torch_to_numpy(v) for v in value])
            else:
                raise ValueError(f"Unsupported type {type(value)}")

            for i, v in enumerate(split_value):
                out_list[i][key] = v

        return out_list
    return wrapper
