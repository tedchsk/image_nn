# from https://pytorch.org/hub/pytorch_vision_resnet/
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from core.model.model_abc import ModelABC


class BasicBlock(nn.Module):

    def __init__(self, inplanes, growthrate: int = 8, stride=1):
        super().__init__()

        # This is one different from ResNet
        self.conv1 = nn.Conv2d(inplanes, growthrate,
                               stride=stride, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(growthrate)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inplanes + growthrate, growthrate,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(growthrate)

        # Change started
        self.block1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.block2 = nn.Sequential(self.conv2, self.bn2)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # Change started
        out = torch.cat((self.block1(x), identity), 1)
        out = torch.cat((self.block2(out), out), 1)
        return self.relu(out)


class TransitionBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super().__init__()

        # This is one different from ResNet
        self.conv1 = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Change started
        self.transition = nn.Sequential(self.conv1, self.avgpool)

    def forward(self, x: Tensor) -> Tensor:
        return self.transition(x)


class DenseNet(ModelABC):
    def __init__(self, model_n, num_classes: int = 10, device=torch.device("cpu")):
        super().__init__()

        self.residual_layers = nn.ModuleList([])
        self.model_n = model_n
        self.device = device
        self.growthrate = 8

        # begining layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # ResNet blocks [16, 32, 64]
        # first block, 16 channels
        in_channels = 16
        for i in range(self.model_n):
            self.residual_layers.append(BasicBlock(
                in_channels, self.growthrate).to(device))
            in_channels += (self.growthrate * 2)
            # Multiplying growthrate by 2 because each iteration adds two basic block layers.

        # second block, 32 channels
        new_in_channels = 32
        for i in range(self.model_n):
            if i == 0:
                self.residual_layers.append(
                    TransitionBlock(in_channels, new_in_channels).to(device))
                in_channels = new_in_channels

            else:
                self.residual_layers.append(BasicBlock(
                    in_channels, self.growthrate).to(device))
                in_channels += (self.growthrate * 2)

        # third block, 64 channels
        new_in_channels = 64
        for i in range(self.model_n):
            if i == 0:
                self.residual_layers.append(
                    TransitionBlock(in_channels, new_in_channels).to(device))
                in_channels = new_in_channels
            else:
                self.residual_layers.append(
                    BasicBlock(in_channels, self.growthrate).to(device))
                in_channels += (self.growthrate * 2)

        # output layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:

        # begining layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet blocks
        for i, layer in enumerate(self.residual_layers):
            print(i, x.shape)
            x = layer(x)

        print(x.shape)
        # output layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
