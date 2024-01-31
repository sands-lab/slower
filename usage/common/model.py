import torch
import torch.nn as nn


def test_accuracy(client_model, server_model, dataloader):
    corrects = 0
    client_model.eval()
    server_model.eval()
    for images, labels in dataloader:
        with torch.no_grad():
            embs = client_model(images)
            preds = server_model(embs)
        preds = torch.argmax(preds, axis=1)
        corrects += (preds == labels).int().sum()
    return corrects / len(dataloader.dataset)


def train(client_model, server_model, dataloader):
    client_model.train()
    server_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        list(client_model.parameters()) + list(server_model.parameters()),
        lr=0.05
    )

    for images, labels in dataloader:
        embs = client_model(images)
        preds = server_model(embs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def reset_parameters(module):
    for param in module.parameters():
        if len(param.shape) > 1:  # Check if parameter is not bias
            nn.init.xavier_uniform_(param)
        else:
            nn.init.constant_(param, 0.0)  # For biases, initialize to zero
