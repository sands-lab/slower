# Slower

Slower is a Split Learning framework built upon Flower.

I suppose that the first two letters in `flower` stand for Federated Learning, so this project is called "Slower", though is might not be the best marketing name.

**NOTE:** I had to manually copy-paste some code from Flower to this repository for typing reasons. For instance, the `maybe_call_evaluate`, `maybe_call_fit`, `EvaluateResultsAndFailures`, and some other pieces of code, could be slightly changed in the flower code and be re-used in this repository. I leave this "clean-up" for future work (either way, it is useful only if afterwards we want to integrate `slower` into `flower`).

## TO-DOs

* IMPROVE GRPC: how should we run the server model segment? Having each server model as a separate ray actor is prohibitively slow!!! Each model run in a separate thread (but what does GIL imply in such a case?)? Run each model in a separate process?? Think about it...;
* Standardize the client_proxy.fit for GRPC and ray implementation (in server.py, there is an if statement to handle kwargs, but it is really ugly!!!)
* Allow the user to specify different models for the server-side segment of the model. This might be useful for instance if there is a high-capacity and a low-capacity device: in such a case, the high capacity device might wish to train a larger portion of the model locally so as to increase the privacy guarantees.
* structure better the imports.
* create a "numpy" version of the raw client/server segment, similar to what happens in flower.
* remove `cid` from proto... thought I will need the information, but I don't.

For the simulation environment build upon ray:

* allow different resources for the server trainer;

Further design questions:

* during evaluation, how should the server behave? Should we have multiple server trainers, or should one server trainer be spanned? Or should we allow the client to customize this?


## How to use the framework?

In the `usage` folder you will find mini-projects that demonstrate in a very simplified way how to use the framework.
