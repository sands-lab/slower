import torch
import torch.nn as nn

from usage.common.model import reset_parameters
from usage.common.helper import seed


_layers = [
    nn.Conv2d(3, 32, 3, 2, 1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 2, 1),
    nn.ReLU(),
    nn.Conv2d(64, 96, 3, 2, 1),
    nn.ReLU(),
    nn.Conv2d(96, 128, 3, 2, 1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(128 * 4, 10)
]


def get_client_model(n_trainable_client_layers):
    assert n_trainable_client_layers < 4, "At most 4 layers can be trained on the client"
    model = nn.Sequential(*_layers[:n_trainable_client_layers])
    seed()
    reset_parameters(model)
    return model


def get_server_model(n_trainable_client_layers):
    assert n_trainable_client_layers < 4, "At most 4 layers can be trained on the client"
    model =  nn.Sequential(*_layers[n_trainable_client_layers:])
    # seed()
    reset_parameters(model)
    return model


def test_accuracy(client_model, server_model, dataloader, device):
    corrects = 0
    client_model.eval()
    server_model.eval()
    server_model.to(device)
    client_model.to(device)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            embs = client_model(images)
            preds = server_model(embs)
        preds = torch.argmax(preds, axis=1)
        corrects += (preds == labels).int().sum()
    return corrects / len(dataloader.dataset)


def train(client_model, server_model, dataloader, lr, device):
    client_model.train()
    server_model.train()
    server_model.to(device)
    client_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        list(client_model.parameters()) + list(server_model.parameters()),
        lr=lr
    )

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        embs = client_model(images)
        preds = server_model(embs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
