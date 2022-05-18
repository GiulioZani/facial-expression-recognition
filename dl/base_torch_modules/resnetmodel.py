from types import SimpleNamespace

import ipdb
import torch
import torch as t
import torch.nn.functional as F
import torchvision

from numpy import prod
from torch import nn, optim

from .dense_layer import AVGPoolConcatDenseLayer
from .conv2dmodel import GaussianNoise


class ResNetBlock(nn.Module):
    def __init__(
        self, input_count, activation, output_reduction=False, output_count=-1
    ):
        """
        input_count - number of input features
        activation - activation function
        output_reduction - reduce output shape by 2 on each axis
        output_count - number of output features
        """
        super().__init__()
        if not output_reduction:
            output_count = input_count

        # network
        self.net = nn.Sequential(
            nn.Conv2d(
                input_count,
                output_count,
                kernel_size=3,
                padding=1,
                stride=1 if not output_reduction else 2,
                bias=False,
            ),
            nn.BatchNorm2d(output_count),  # handles bias
            activation(),
            nn.Conv2d(
                output_count,
                output_count,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_count),
            nn.Dropout(0.3),
        )

        # 1x1 convolution, stride 2
        self.downsample = (
            nn.Conv2d(input_count, output_count, kernel_size=1, stride=2)
            if output_reduction
            else None
        )
        self.activation = activation()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.activation(out)
        return out


class PreActResNetBlock(nn.Module):
    def __init__(
        self, input_count, activation, output_reduction=False, output_count=-1
    ):
        """
        input_count - number of input features
        activation - activation function
        output_reduction - reduce output shape by 2 on each axis
        output_count - number of output features
        """
        super().__init__()
        if not output_reduction:
            output_count = input_count

        # network
        self.net = nn.Sequential(
            nn.BatchNorm2d(input_count),
            activation(),
            nn.Conv2d(
                input_count,
                output_count,
                kernel_size=3,
                padding=1,
                stride=1 if not output_reduction else 2,
                bias=False,
            ),
            nn.BatchNorm2d(output_count),
            activation(),
            nn.Conv2d(
                output_count,
                output_count,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

        # 1x1 convolution, non-linearity | not done on skip connection
        self.downsample = (
            nn.Sequential(
                nn.BatchNorm2d(input_count),
                activation(),
                nn.Conv2d(
                    input_count,
                    output_count,
                    kernel_size=1,
                    stride=2,
                    bias=False,
                ),
            )
            if output_reduction
            else None
        )

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out


resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock,
}
activation_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class ResNetEmotionClassifier(nn.Module):
    def __init__(
        self,
        params,
        blocks_count=[3, 3, 3],
        blocks_dimensions=[32, 64, 128],
        activation_name="relu",
        block_name="ResNetBlock",
        output_block="avgpool_plus_dense",
        **kwargs
    ):
        """
        blocks_count - numbers of ResNet blocks to use
        blocks_dimensions - dimensionalities in different blocks
        activation_name - activation function to use
        block_name - ResNet block
        output_block - last block in the network generating the output
        """
        super().__init__()
        self.params = params
        self.output_block = output_block

        assert block_name in resnet_blocks_by_name
        self.hparams = SimpleNamespace(
            blocks_dimensions=blocks_dimensions,
            blocks_count=blocks_count,
            activation_name=activation_name,
            activation=activation_by_name[activation_name],
            block_class=resnet_blocks_by_name[block_name],
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        blocks_dimensions = self.hparams.blocks_dimensions

        # convolution on original image (upscaling channel size)
        if (
            self.hparams.block_class == PreActResNetBlock
        ):  # only apply non-linearity on non-outputs
            self.input_net = nn.Sequential(
                nn.Conv2d(
                    1,
                    blocks_dimensions[0],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(
                    1,
                    blocks_dimensions[0],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(blocks_dimensions[0]),
                self.hparams.activation(),
            )

        # creating ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.blocks_count):
            for bc in range(block_count):
                output_reduction = (
                    bc == 0 and block_idx > 0
                )  # output_reduction on first block of each group (except in first one)
                blocks.append(
                    self.hparams.block_class(
                        input_count=blocks_dimensions[
                            block_idx
                            if not output_reduction
                            else (block_idx - 1)
                        ],
                        activation=self.hparams.activation,
                        output_reduction=output_reduction,
                        output_count=blocks_dimensions[block_idx],
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(blocks_dimensions[-1], 8),
        )

    def _init_params(self):
        # initialize convolutions according to activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity=self.hparams.activation_name,
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = x.squeeze(2)

        # add noise:
        # noise_matrix_size = 12
        # noise = t.randn( self.params['generator_in_seq_len'] , noise_matrix_size, noise_matrix_size)

        # x_concat = t.zeros((x.shape[0], x.shape[1], x.shape[2]+ noise_matrix_size, x.shape[3]+ noise_matrix_size))
        # x_concat[:,:,:x.shape[2], :x.shape[3]] = x
        # x_concat[:,:,x.shape[2]:, x.shape[3]:] = noise

        x = self.input_net(x)
        x = self.blocks(x)

        x = self.output_net(x)
        return x
