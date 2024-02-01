import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CustomDataset(Dataset):
    def __init__(self, images_folder, partition_csv, transforms=None, metadata=None):
        super().__init__()
        self.df = pd.read_csv(partition_csv)
        if metadata is not None:
            assert len(metadata) == self.df.shape[0], \
                f"Not matching shapes: {metadata.shape} {self.df.shape}"
            assert isinstance(metadata, np.ndarray)
            self.metadata = metadata
        else:
            self.metadata = None
        self.images_folder = images_folder
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row["label"]
        image_file = row["filename"]
        image = Image.open(f"{self.images_folder}/{image_file}")
        if self.transforms:
            image = self.transforms(image)
        if self.metadata is None:
            return image, target
        mtd = torch.from_numpy(self.metadata[idx])
        return image, target, mtd


class UnlabeledDataset(Dataset):
    def __init__(self, dataset_name, dataset_size) -> None:
        super().__init__()
        data_home_folder = os.environ.get("FLTB_DATA_HOME_FOLDER")
        dataset_home_folder = f"{data_home_folder}/{dataset_name}"
        self.filepaths = pd.read_csv(f"{dataset_home_folder}/metadata.csv")["filename"]\
            .sample(dataset_size, replace=False).to_list()
        self.dataset_home_folder = dataset_home_folder
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(f"{self.dataset_home_folder}/{self.filepaths[idx]}")
        image = self.to_tensor(image)
        return image
