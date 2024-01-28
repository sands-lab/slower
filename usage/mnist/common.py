import torch
import torch.nn as nn
import torchvision


def get_dataloader():
    dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            nn.Flatten(start_dim=0)
        ])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader
