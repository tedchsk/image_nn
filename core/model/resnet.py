import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

# from https://pytorch.org/hub/pytorch_vision_resnet/


class _BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, down=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None

        if down:
            self.downsample = nn.Conv2d(
                inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, model_n, num_classes: int = 10, device=torch.device("cpu")):
        super().__init__()

        self.residual_layers = nn.ModuleList([])
        self.model_n = model_n

        # begining layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # ResNet blocks [16, 32, 64]
        # first block, 16 channels
        for i in range(self.model_n):
            self.residual_layers.append(_BasicBlock(16, 16).to(device))

        # second block, 32 channels
        for i in range(self.model_n):
            if i == 0:
                self.residual_layers.append(_BasicBlock(
                    16, 32, stride=2, down=True).to(device))
            else:
                self.residual_layers.append(_BasicBlock(32, 32).to(device))

        # third block, 64 channels
        for i in range(self.model_n):
            if i == 0:
                self.residual_layers.append(_BasicBlock(
                    32, 64, stride=2, down=True).to(device))
                self.inplanes = 64
            else:
                self.residual_layers.append(_BasicBlock(64, 64).to(device))

        # output layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:

        # begining layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet blocks
        for i, layer in enumerate(self.residual_layers):
            x = layer(x)

        # output layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
