# from https://pytorch.org/hub/pytorch_vision_resnet/
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from core.model.model_abc import ModelABC


class BasicBlock(nn.Module):
    """Basic DSNet. Given input [in_channels, height, width], 
    - First pass through Conv2d(in_channels, in_channels) + BatchNorm + ReLU 
        -> Output dimensions: [in_channels, height, width] (1)
    - Then, add with the ("normalization and channel-wise weight")(input)
        -> Output dimensions: [in_channels, height width] (2)
    - Pass through another Conv2d(outchannels, outchannels) + BN + ReLU
        -> Output dimensions: [in_channels, height, width] (3)
    - Add again with ("normalized + channel-wise weight")(1) and ("normalized + channel-wise weight")(2)
        -> Output dimensions: [in_channels, height, width]

    Caveat: The normalization and channel-wise weight is not shared.

    Attributes:
        in_planes: # of Input channels
        n_models: Number of layers. Have to specify here as we need to connect all the layers
    """

    def __init__(self, inplanes, n_models, device=torch.device("cpu")):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.channel_wise_w_list = []  # Result is list of list of weights at each steps
        for i in range(n_models * 2):
            self.layers.append(nn.Sequential(
                nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1),
                nn.BatchNorm2d(inplanes)
            ))

            # One variable for each channel for each time, [[w00], [w10, w11], [w20, w21, w22], ...]
            self.channel_wise_w_list.append(
                [torch.autograd.Variable(torch.randn(1, inplanes, 1, 1).to(device), requires_grad=True)
                 for _ in range(i+1)]
            )

        self.normalization = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.normalization(x)
        # Consisting of output of each layer.
        outputs = [identity]
        for (layer, ch_ws) in zip(self.layers, self.channel_wise_w_list):
            output = layer(outputs[-1])

            assert len(outputs) == len(ch_ws), "Length not equal"
            dense_normalized_inputs = [x * ch_weight
                                       for output, ch_weight in zip(outputs, ch_ws)]
            for dense_normalized_input in dense_normalized_inputs:
                output += dense_normalized_input

            output = self.relu(output)
            outputs.append(output)

        return outputs[-1]


class TransitionBlock(nn.Module):
    """A transition block to reduce channels of [input, w, h] to [outplanes, w//2, h//2]

    Attributes:
        inplanes: # of Input channels
        outplanes: # of Output channels
    """

    def __init__(self, inplanes, outplanes):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inplanes, outplanes, stride=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.block = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.conv2, self.bn2)

        self.downsample = nn.Conv2d(
            inplanes, outplanes, kernel_size=1, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.block(x) + identity
        return self.relu(out)


class DSNet(ModelABC):
    """Defining the whole model. 
    In high level: 
        - Input -> [batch, 3, height, width]
        - Beginning Layer -> [batch, 3, height, width]

        - First Block: n*BasicBlock(16) -> [batch, 16, height, width]
        - Transition: TransitionBlock(16, 32) -> [batch, 32, height, width]

        - Second Block: n*BasicBlock(32) -> [batch, 32, height, width]
        - Transition: TransitionBlock(32, 64) -> [batch, 32, height, width]

        - Third Block: n*BasicBlock(64) -> [batch, 64, height, width]

        - FinalLayer: AdaptiveAvgPool2d + Linear(64, num_classes)

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

        # begining layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # ResNet blocks [16, 32, 64]
        # first block, 16 channels
        self.residual_layers.append(BasicBlock(
            16, self.model_n, device).to(device))
        self.residual_layers.append(TransitionBlock(16, 32).to(device))

        # second block, 32 channels
        self.residual_layers.append(BasicBlock(
            32, self.model_n, device).to(device))
        self.residual_layers.append(TransitionBlock(32, 64).to(device))

        # third block, 64 channels
        self.residual_layers.append(BasicBlock(
            64, self.model_n, device).to(device))
        self.residual_layers.append(TransitionBlock(64, 64).to(device))

        # output layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # begining layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # DSNet blocks
        for i, layer in enumerate(self.residual_layers):
            x = layer(x)

        # output layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
