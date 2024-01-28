# Slower

Slower is a Split Learning framework built upon Flower.

I suppose that the first two letters in `flower` stand for Federated Learning, so this project is called "Slower", though is might not be the best marketing name.

**NOTE:** I had to manually copy-paste some code from Flower to this repository for typing reasons. For instance, the `maybe_call_evaluate`, `maybe_call_fit`, `EvaluateResultsAndFailures`, and some other pieces of code, could be slightly changed in the flower code and be re-used in this repository. I leave this "clean-up" for future work (either way, it is useful only if afterwards we want to integrate `slower` into `flower`).

## TO-DOs

* Implement GRPC;
* Allow the user to specify different models for the server-side segment of the model. This might be useful for instance if there is a high-capacity and a low-capacity device: in such a case, the high capacity device might wish to train a larger portion of the model locally so as to increase the privacy guarantees.
* structure better the imports.

For the simulation environment build upon ray:

* allow different resources for the server trainer;


## How to use the framework?

In the `usage` folder you will find mini-projects that demonstrate in a very simplified way how to use the framework.
