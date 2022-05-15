from ...base_lightning_modules.base_model import BaseModel

import torch as t
from torch import nn
import torch.nn.functional as F
from argparse import Namespace


class ConvModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return F.relu(self.conv1(x))


class Model(BaseModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = ConvModel(params)
