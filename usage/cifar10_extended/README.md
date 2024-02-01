# Testing slower on CIFAR10

The model used in this experiment is a simple convolutional network in only one fully connected layer.

**Note**: in order to run this test, you need to first generate the partitioning with the `dcml_algorithms` framework. After you generate the data partitioning, fill update the `.env` file with the appropriate data.

## Benchmarking

Approximate obtained values:

|                    | FL    | SL        |
|--------------------|-------|-----------|
| total running time | 204.4 | 208.7     |
| accuracy           | 46.8% | 47.9%     |

Results were obtained with the following configuration:

```python
N_CLIENT_LAYERS=2
N_EPOCHS=20
CLIENT_RESOURCES={"num_cpus": 4, "num_gpus": 0.25}
COMMON_SERVER=False
LR=0.05
FRACTION_FIT=0.25
```
