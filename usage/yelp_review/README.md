# Testing slower on Yelp Reviews

The model used in this experiment is a pre-trained Bert model. We train the full model, even though we might restrict to training only the last fully connected layer and hence significantly reduce the requirements to the client.

## Benchmarking

Approximate running times for $4$ training epochs ($4$ server rounds in the case of SL/FL, as in each server round we perform $1$ training epoch):

- Centralized training: $103s$;
- SL: $304s$ (because of the compression/decompression, time for serialization is almost negligible. Roughly $0.12$ during training and $0.8$ during evaluation. Total $0.2 * 2 * 4 = 2s$);
- FL: $312s$.

Results obtained by setting `client_resources={"num_cpus": 4, "num_gpus": 0.}` (no constraints on the centralized version), and by using a dataset with only $64$ training/evaluation examples.
