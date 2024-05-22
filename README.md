# Slower

Slower is a Split Learning framework built upon Flower.

I suppose that the first two letters in `flower` stand for Federated Learning, so this project is called "Slower", though is might not be the best marketing name.

**NOTE:** I had to manually copy-paste some code from Flower to this repository for typing reasons. For instance, the `maybe_call_evaluate`, `maybe_call_fit`, `EvaluateResultsAndFailures`, and some other pieces of code, could be slightly changed in the flower code and be re-used in this repository. I leave this "clean-up" for future work (either way, it is useful only if afterwards we want to integrate `slower` into `flower`).

## TO-DOs

* IMPROVE GRPC: how should we run the server model segment? Having each server model as a separate ray actor is prohibitively slow!!! Each model run in a separate thread (but what does GIL imply in such a case?)? Run each model in a separate process?? Think about it...;
* Improve simulation when clients share a common server - currently I use a Ray actor to ensure consistency, but this solution is very slow!!
* Ensure that manipulation of the manipulation of the random seed within the client code of the server model segment has no effect to the strategy. Otherwise, if in the server model segment code we set the seed, it might alter the behaviour of the strategy and in the worst case cause the sampling of the very same clients in all the epochs;
* Allow the user to specify different models for the server-side segment of the model. This might be useful for instance if there is a high-capacity and a low-capacity device: in such a case, the high capacity device might wish to train a larger portion of the model locally so as to increase the privacy guarantees.
* structure better the imports.
* when running on ray, allow different resources for the server trainer (this is strictly connected with point 2. above);

Further design questions:

* during evaluation, how should the server behave? Should we have multiple server trainers, or should one server trainer be spanned? Or should we allow the client to customize this?


**NOTE**: currently, the client is expected to send to the server bytes. Of course, in future we want to change this and let the client to send to the server a numpy array and let the framework serialize the data. However, for now I want to have control over the serialization in the client code, so that we can experiment with different serialization approaches.


## How to use the framework?

In the `usage` folder you will find mini-projects that demonstrate in a very simplified way how to use the framework.


## Installation

```bash
pip install -U -I git+ssh://git@github.com/sands-lab/slower@master
# or
pip install -U -I git+https://git@github.com/sands-lab/slower@master
```
