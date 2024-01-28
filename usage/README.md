# Purpose

In this folder are present a number of toy-projects that aim to verify:

1. That the `slower` framework works correctly. To verify this, we remove all sources of uncertaintly (e.g., we set the seed of the RND, we do not shuffle the data in the dataloader, ...). Therefore, the accuracy of running the experiment with the SL framework should be the very same as the ones when running the the training in a fully-centralized manner;
2. The performance of the `slower` framework as compared to the when using the `flower` framework and the fully-centralized manner. Note, that when using separate server model segments for every client, SL is actually equivalent to FL (that is, it should give the very same results, even though a part of the computation is performed on the server). Note, that `slower` is expected to be to some extent slower than `flower` because of the serialization-deserialization process happening for every batch of data both on the server and the client.

Therefore, each sub-project contains the following files:

- `fl.py`: plain implementation of training with FedAvg using Flower.
- `centralized.py`: simulation of training in a fully centralized way.
- `sl.py`: simulation of t

To summarize, if everything works ok and you are using separate server model segments, all three scripts should yield the **very same accuracies**. Regarding the times, the fully centralized training is expected to be the fastest (no communication, no synchronization, not limiting resources), followed by the FL simulation and lastly the SL simulation.
