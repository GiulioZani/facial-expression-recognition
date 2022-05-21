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
import numpy as np
import sys
import matplotlib.pyplot as plt


class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.data_location = params.data_location
        self.params = params
        self.shuffle = True
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
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
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
        image = x.numpy()
        labels = self.labels[idx].long()
        # augmentation_prob = 0.15
        augmentation_prob = 0.1
        pick = t.rand(1).item()
        # determine augmentation category:
        if pick < augmentation_prob:  # Sharpening filter
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image_sharp = t.from_numpy(
                cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
            )
            return image_sharp, labels
        elif pick < 2 * augmentation_prob:  # Image Rotation
            image_filtered = t.from_numpy(
                cv2.GaussianBlur(image, (5, 5), sigmaX=0)
            )
            return image_filtered, labels
        elif pick < 3 * augmentation_prob:  # Image Rotation
            # calculate center of image
            h, w = image.shape[:2]
            cX, cY = (w // 2, h // 2)
            # get rotation matrix
            M_rotate_x_45 = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
            # rotate by 45 degrees
            image_rotated_45 = t.from_numpy(
                cv2.warpAffine(image, M_rotate_x_45, (w, h))
            )
            return image_rotated_45, labels
        else:
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
