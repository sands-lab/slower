# Testing slower on CIFAR10

The model used in this experiment is a simple convolutional network in only one fully connected layer.


## Benchmarking

Approximate running times for $4$ training epochs ($4$ server rounds in the case of SL/FL, as in each server round we perform $1$ training epoch):

- Centralized training: $123s$ ();
- SL: $300s$ ($1.6s$ for serialization/deserialization during training, $1.3$ for serialization/deserialization during evaluation. This happens on both the client and the server. So the expected overhead is $(1.6+1.3) * 2 * 4=23s$);
- FL: $50s$.

Results obtained by setting `client_resources={"num_cpus": 2, "num_gpus": 0.}`.
