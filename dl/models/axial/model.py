from torch.nn.modules.activation import ReLU
from ...base_lightning_modules.base_classification_model import (
    BaseClassificationModel,
)
import ipdb

from ...base_torch_modules.resnetmodel import ResNetEmotionClassifier
import torch as t
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
from axial_attention import AxialAttention, AxialPositionalEmbedding


class Axial(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        imsize = (params.imsize, params.imsize)
        self.embedding = AxialPositionalEmbedding(dim=1, shape=imsize)
        self.attention = AxialAttention(dim=8, dim_index=1, heads=8)
        self.linear = nn.Linear(params.imsize ** 2, 1)

    def forward(self, x):
        x = x.repeat(1, 8, 1, 1)
        return self.linear(
            self.attention(self.embedding(x)).view(x.shape[0], x.shape[1], -1)
        ).squeeze(-1)


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = Axial(params)
