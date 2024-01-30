# Testing slower on CIFAR10

The model used in this experiment is a simple convolutional network in only one fully connected layer.


## Benchmarking

Approximate running times for $4$ training epochs ($4$ server rounds in the case of SL/FL, as in each server round we perform $1$ training epoch):

|                    | centralized | FL   | SL (Ray) | SL (gRPC) | Serialization+deserialization in SL (train/eval) |
|--------------------|-------------|------|----------|-----------|--------------------------------------------------|
| total running time | 13          | 22.3 | 27.7     | 30.2      | 0.35/0.30                                        |

Results were obtained with the following configuration:

```python
N_EPOCHS=4
N_CLIENTS=2  # needs to be at least 2, otherwise FL will hang
CLIENT_RESOURCES={"num_cpus": 4, "num_gpus": 0.0}
COMMON_SERVER=False  # for split learning
USE_NUMPY_CLIENTS=False  # whether to use the numpy version of the clients or not
```

*Note*: The gRPC results were obtained with only 1 client.
