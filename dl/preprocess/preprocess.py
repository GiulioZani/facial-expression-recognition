import torch as t
import os
import h5py
import ipdb
from tqdm import tqdm
from ..utils.data_manager import DataManger
import h5py
import cv2
import pandas as pd


def extract(file_names, imsize: int):
    return tuple(
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


def preprocess(
    data_location: str = "/mnt/facial_expression_dataset",
    imsize: int = 128,
    destination_folder: str = "/mnt/preprocessed_data.h5",
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
    train_data = extract(train_file_names, imsize)
    test_data = extract(test_file_names, imsize)
    # saves train_data and test_data in h5py format
    with h5py.File(destination_folder, "w") as f:
        ipdb.set_trace()
        f.create_dataset("train", data=train_data)
        f.create_dataset("test", data=test_data)
