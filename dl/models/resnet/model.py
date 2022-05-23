from torch.nn.modules.activation import ReLU
from ...base_lightning_modules.base_classification_model import BaseClassificationModel
import ipdb

from ...base_torch_modules.resnetmodel import ResNetEmotionClassifier
import torch as t
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
import matplotlib.pyplot as plt


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = ResNetEmotionClassifier(params)

    def test_step(self, batch, batch_nb):
        if False:
            x, y = batch
            act = self.generator(x, act=True)

            class_labels = [
                "sad",
                "surprise",
                "neutral",
                "happy",
                "disgust",
                "contempt",
                "anger",
                "fear",
            ]
            _, axes = plt.subplots(x.shape[0], 2)
            for i in range(x.shape[0]):
                label = class_labels[y[i].int().item()]
                axes[i][0].imshow(x[i, 0].cpu())
                axes[i][1].imshow(act[i].cpu())
                axes[i][0].set_title(f"{label}")
                axes[i][1].set_title(f"{label}")
            plt.show()
        super().test_step(batch, batch_nb)

