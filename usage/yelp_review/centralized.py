import time

import torch

from common import get_dataloader
from constants import N_EPOCHS
from transformers import AutoModelForSequenceClassification


from usage.common.helper import seed
from usage.common.model import reset_parameters


def train(model, trainloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for batch in trainloader:
        out = model(**batch)
        loss = out.loss

        model.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, testloader):
    n_corrects = 0

    for batch in testloader:
        with torch.no_grad():
            logits = model(**batch).logits
        predictions = torch.argmax(logits, dim=-1)
        n_corrects += (predictions == batch["labels"]).sum().item()

    print(f"Accuracy: {n_corrects / len(testloader.dataset):.3f}")


def main():
    start = time.time()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    seed()
    reset_parameters(model.bert.pooler)
    reset_parameters(model.classifier)

    model.eval()  # this way the dropout layers behave deterministically
    dataloader = get_dataloader()
    for _ in range(N_EPOCHS):
        train(model, dataloader)
        evaluate(model, dataloader)

    print(f"Total time: {time.time() - start}")


if __name__ == "__main__":
    main()
