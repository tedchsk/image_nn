# from https://pytorch.org/hub/pytorch_vision_resnet/
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from core.model.model_abc import ModelABC


class BasicBlock(nn.Module):
    """Basic DenseBlock. Given input [in_channels, height, width], 
    - First pass through Conv2d(in_channels, outchannels=growthrate) + BatchNorm + ReLU 
        -> Output dimensions: [growthrate, height, width] (1)
    - Then, concatenate with the input
        -> Output dimensions: [in_channels + growthrate, height, width] (2)
    - Pass through another Conv2d(in_channels + growthrate) + BN + ReLU
        -> Output dimensions: [growthrate, height, width] (3)
    - Concatenate again to (2) get the output
        -> Output dimensions: [(in_channels + growthrate) + growthrate, height, width],

    Caveat: One basic block contains two DenseLayers, so the output channels will be in_channels + 2*growthrate

    Attributes:
        inplanes: # of Input channels
        growthrate: Additional channels we'll get from each DenseLayer.
    """

    def __init__(self, inplanes, growthrate: int = 8):
        super().__init__()

        # This is one different from ResNet (kernel=1 no padding as opposed to kernel=3 & padding=1)
        self.conv1 = nn.Conv2d(inplanes, growthrate, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(growthrate)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inplanes + growthrate, growthrate,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(growthrate)

        self.block1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.block2 = nn.Sequential(self.conv2, self.bn2)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # Concatenate instead of adding.
        out = torch.cat((self.block1(x), identity), 1)
        out = torch.cat((self.block2(out), out), 1)
        return self.relu(out)


class TransitionBlock(nn.Module):
    """A transition block to reduce channels of [input + growthrate * n, w, h] to [new_input_channels, w, h]

    Attributes:
        inplanes: # of Input channels
        outplanes: # of Output channels
    """

    def __init__(self, inplanes, outplanes):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.transition = nn.Sequential(self.conv1, self.avgpool)

    def forward(self, x: Tensor) -> Tensor:
        return self.transition(x)


class DenseNet(ModelABC):
    """Defining the whole model. 
    In high level: 
        - Input -> [batch, 3, height, width]
        - Beginning Layer -> [batch, 3, height, width]

        - First Block: n*BasicBlock(16) -> [batch, 16 + 2n * growthrate, height, width]
        - Transition: TransitionBlock(16 + 2n * growthrate, 32) -> [batch, 32, height, width]

        - Second Block: n*BasicBlock(32) -> [batch, 32 + 2n * growthrate, height, width]
        - Transition: TransitionBlock(32 + 2n * growthrate, 64) -> [batch, 32, height, width]

        - Third Block: n*BasicBlock(16) -> [batch, 64 + 2n * growthrate, height, width]

        - FinalLayer: AdaptiveAvgPool2d + Linear(64 + 2n * growthrate, num_classes)

    Attributes:
        model_n: # of layers, based on CIFAR-ResNet 
        num_classes: Number of classes
        device: needed for GPU vs CPU.
    """

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
            x = layer(x)

        # output layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
