# Testing slower on Mnist

The model used in this experiment is composed on only fully connected layers. Specifically, the input images are flattened to get vectors with size `28 * 28`, and then the model is composed of three hidden layers (in SL, the first layer is trained on the client and the remaining two on the server).

## Benchmarking

Approximate running times for $4$ training epochs ($4$ server rounds in the case of SL/FL, as in each server round we perform $1$ training epoch):

- Centralized training: $45s$;
- SL: $60s$ ($0.7s$ for serialization/deserialization during training, $0.5$ for serialization/deserialization during evaluation. This happens on both the client and the server. So the expected overhead is $(0.7+0.5) * 2 * 4=9.6s$);
- FL: $50s$.

Results obtained by setting `client_resources={"num_cpus": 2, "num_gpus": 0.}`.
