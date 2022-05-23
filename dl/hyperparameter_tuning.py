import sklearn
import optuna
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import ipdb
import skvideo.io
import torch as t

from models.resnet.model import Model


def run_tuning():
    study = optuna.create_study()  # create process
    study.optimize(objective, n_trials=100)  # optimize


def objective(trial):
    # choose network parameters:
    # "cuda": true,
    # "train_batch_size": 16,
    # "test_batch_size": 50,
    # "imsize": 128,
    # "n_channels": 4,
    # "ngf": 64,
    # "ndf": 32,
    # "epochs": 10,
    # "lr": 0.0002,
    # "b1": 0.5,
    # "b2": 0.999,
    # "gaussian_noise_std": 0.1,
    # "in_seq_len": 5,
    # "out_seq_len": 5,
    # "max_epochs": 100,
    # "early_stopping_patience": 5,
    # "reduce_lr_on_plateau_patience": 2,
    # "data_location": "/mnt/preprocessed_data.h5",
    # "crop":64

    # trial.suggest_categorical('name', ['opt1', 'opt2'])
    # trial.suggest_int('name', 2, 32, step=2w)

    # create network:

    # train network:

    # test performance:
    y_val = []
    y_predicted = []

    # return score to be minimized during tuning:
    return sklearn.metrics.mean_squared_error(y_val, y_predicted)


if __name__ == "__main__":
    global_model = Model
    global_data = skvideo.io.vread("videos/video4.mp4")[::1]
    global_face_classifier = cv2.CascadeClassifier(
        "dl/video/face_detector/haarcascade_frontalface_default.xml"
    )
    global_class_labels = [
        "sad",
        "surprise",
        "neutral",
        "happy",
        "disgust",
        "contempt",
        "anger",
        "fear",
    ]
    run_tuning()
