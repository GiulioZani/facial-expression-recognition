import torch as t
import os
import h5py
import ipdb
from tqdm import tqdm
from ..utils.data_manager import DataManger
import h5py
import cv2
import pandas as pd
from argparse import ArgumentParser


def extract(file_names, imsize: int):
    data_with_labels = tuple(
        (
            label,
            t.from_numpy(
                cv2.resize(
                    cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY),
                    (imsize, imsize),
                )
            )
            / 255,
        )
        for file_name, label in file_names
    )
    labels = t.tensor([label for label, _ in data_with_labels])
    data = t.stack([data for _, data in data_with_labels])
    return labels, data


def preprocess(
    data_location: str = "data",
    imsize: int = 128,
    destination_folder: str = "preprocessed_data.h5",
):
    labels = {
        key: val
        for key, val in pd.read_csv(
            os.path.join(data_location, "labels.csv")
        ).values.tolist()
    }
    emotion_keys = {
        val: key for key, val in enumerate(tuple(set(labels.values())))
    }
    emotion_keys = emotion_keys
    labels = {key: emotion_keys[val] for key, val in labels.items()}
    files = tuple(
        (os.path.join(data_location, file_name), label)
        for file_name, label in labels.items()
    )
    rand_indices = t.randperm(len(files))
    shuffled_files = tuple(files[i] for i in rand_indices)
    cutting_index = int(0.2 * len(files))
    test_file_names = shuffled_files[:cutting_index]
    train_file_names = shuffled_files[cutting_index:]
    train_labels, train_data = extract(train_file_names, imsize)
    test_labels, test_data = extract(test_file_names, imsize)
    # saves train_data and test_data in h5py format
    with h5py.File(destination_folder, "w") as f:
        f.create_dataset("train_data", data=train_data)
        f.create_dataset("train_labels", data=train_labels)
        f.create_dataset("test_data", data=test_data)
        f.create_dataset("test_labels", data=test_labels)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_location",
        type=str,
        default="data",
        help="Folder where dataset is saved.",
    )
    parser.add_argument(
        "--imsize",
        type=int,
        default=128,
        help="Size of the image after downsampling.",
    )
    parser.add_argument(
        "--destination_folder",
        type=str,
        default="preprocessed_data.h5",
        help="Folder where the preprcessed dataset is saved.",
    )
    args = parser.parse_args()
    preprocess(**args.__dict__)
