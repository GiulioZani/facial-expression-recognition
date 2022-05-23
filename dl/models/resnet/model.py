from torch.nn.modules.activation import ReLU
from ...base_lightning_modules.base_classification_model import BaseClassificationModel
import ipdb

from ...base_torch_modules.resnetmodel import ResNetEmotionClassifier
import torch as t
from torch import nn
import torch.nn.functional as F
from argparse import Namespace


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = ResNetEmotionClassifier(params)
