# Testing slower on Yelp Reviews

The model used in this experiment is a pre-trained Bert model. We train the full model, even though we might restrict to training only the last fully connected layer and hence significantly reduce the requirements to the client.

## Benchmarking

Approximate running times:

|                    | centralized | FL  | SL (Ray) | SL (gRPC) | Serialization+deserialization in SL (train/eval) |
|--------------------|-------------|-----|----------|-----------|--------------------------------------------------|
| total running time | 134         | 292 | 298      | 168       | 0.15/0.08                                        |

Results were obtained with the following configuration:

```python
N_CLIENT_LAYERS=2
N_EPOCHS=4
N_CLIENTS=2  # needs to be at least 2, otherwise FL will hang
CLIENT_RESOURCES={"num_cpus": 4, "num_gpus": 0.0}
```

*Note*: The gRPC results were obtained with only 1 client. Client and server had no constraints regarding resource usage.
