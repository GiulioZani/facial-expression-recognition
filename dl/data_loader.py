import os
import torch as t
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from .utils.data_manager import DataManger
import pandas as pd
import cv2


class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.data_location = params.data_location
        self.params = params
        # reads the file names from the data_location in h5 format
        with h5py.File(self.data_location, "r") as f:
            self.train_data = t.from_numpy(f["train_data"][:])
            self.train_labels = t.from_numpy(f["train_labels"][:])
            self.test_data = t.from_numpy(f["test_data"][:])
            self.test_labels = t.from_numpy(f["test_labels"][:])

    def train_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(
            self.train_data,
            self.train_labels,
            train=True,
            imsize=self.params.imsize,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=3,
        )

    def val_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(
            self.train_data,
            self.test_labels,
            train=False,
            imsize=self.params.imsize,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=3,
        )

    def test_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(
            self.test_data,
            self.test_labels,
            train=False,
            imsize=self.params.imsize,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=3,
        )


class CustomDataset(Dataset):
    def __init__(
        self, data, labels, train: bool = True, imsize: int = 46,
    ):
        self.data = data
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].unsqueeze(0)
        labels = t.eye(8)[self.labels[idx]].float()
        return x, labels


def test():
    data_module = CustomDataModule()
    train_dataloader = data_module.train_dataloader()
    for i, (x, y) in enumerate(train_dataloader):
        print(i, x.shape, y.shape)
    """
    train_dl, test_dl = get_loaders(
        "/mnt/tmp/multi_channel_train_test",
        32,
        64,
        t.device("cuda" if t.cuda.is_available() else "cpu"),
        in_seq_len=8,
        out_seq_len=4,
    )
    for i, (x, y) in enumerate(tqdm(train_dl)):
        # plt.imshow(x[0, 0, 0].cpu())
        # plt.show()
        # print(x.shape)
        # return
        # print(f"iteration: {i}")
        pass
    """
    # reads file in h5 format


if __name__ == "__main__":
    test()
