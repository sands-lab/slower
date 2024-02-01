import os
import json

import torch
import torchvision.transforms as T

from usage.cifar10_extended.cv_dataset import CustomDataset


def get_dataloader(is_train, cid):
    transforms = [T.ToTensor()]
    if is_train:
        transforms.append(T.RandomHorizontalFlip())
    images_folder = os.getenv("IMAGES_FOLDER")
    partition_folder = os.getenv("PARTITION_FOLDER")
    tmp = "train" if is_train else "test"
    partition_csv = f"{partition_folder}/partition_{cid}_{tmp}.csv"
    dataset = CustomDataset(
        images_folder=images_folder,
        partition_csv=partition_csv,
        transforms=T.Compose(transforms)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=is_train)
    return dataloader


def load_data_config(partition_folder):
    with open(f"{partition_folder}/generation_config.json", "r") as fp:
        data = json.load(fp)
    return data
