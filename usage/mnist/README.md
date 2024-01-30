# Testing slower on Mnist

The model used in this experiment is composed on only fully connected layers. Specifically, the input images are flattened to get vectors with size `28 * 28`, and then the model is composed of three hidden layers (in SL, the first layer is trained on the client and the remaining two on the server).

## Benchmarking

Approximate running times:

|                    | centralized | FL | SL (Ray) | SL (gRPC) | Serialization+deserialization in SL (train/eval) |
|--------------------|-------------|----|----------|-----------|--------------------------------------------------|
| total running time | 44          | 54 | 64       | 65        | 0.75/0.55                                        |

Results were obtained with the following configuration:

```python
N_EPOCHS=4
N_CLIENTS=2  # needs to be at least 2, otherwise FL will hang
CLIENT_RESOURCES={"num_cpus": 4, "num_gpus": 0.0}
COMMON_SERVER=False
```

*Note*: The gRPC results were obtained with only 1 client.
