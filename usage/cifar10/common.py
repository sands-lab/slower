import torch
import torchvision


def get_dataloader():
    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=False,  # use the test dataset to spped up the evaluation
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader
