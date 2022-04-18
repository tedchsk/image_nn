import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

class _DenseBlock(nn.Module):
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

    def __init__(self, planes, n_models, device=torch.device("cpu"), stride=1, down=False, downsample=None):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.channel_wise_w_list = []  # Result is list of list of weights at each steps
        self.norm_layers = nn.ModuleList([])
        self.downsample = downsample
        
        if down:
            inplanes = planes//2
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        else:
            inplanes = planes

        for i in range(n_models):
            if i == 0:
                first_conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride)
            else:
                first_conv = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1)
            
            self.layers.append(nn.Sequential(
                first_conv,
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes)
            ))

            self.norm_layers.append(
                nn.ModuleList([nn.GroupNorm(num_groups=4, num_channels=planes).to(device) for _ in range(i+1)])
            )

            # One variable for each channel for each time, [[w00], [w10, w11], [w20, w21, w22], ...]
            self.channel_wise_w_list.append(
                [nn.Parameter(torch.randn(1, planes, 1, 1).to(device), requires_grad=True)
                 for _ in range(i+1)]
            )
            
            for j, p_list in enumerate(self.channel_wise_w_list):
                for k, p in enumerate(p_list):
                    self.register_parameter("channel_weight_{}_{}".format(j,k), p)
                       
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:


        if self.downsample is not None:
            original_x = x
            x = self.downsample(x)
        
        # Consisting of output of each layer.
        outputs = [x]
        
        for i,(layer, ch_ws, norm_layer) in enumerate\
        (zip(self.layers, self.channel_wise_w_list, self.norm_layers)):
            
            if i==0 and self.downsample is not None:
                output = layer(original_x)
            else:
                output = layer(outputs[-1])

            assert len(outputs) == len(ch_ws), "Length not equal"
            dense_normalized_inputs = [norm(o) * ch_weight
                                       for o, ch_weight, norm in zip(outputs, ch_ws, norm_layer)]
            for dense_normalized_input in dense_normalized_inputs:

                output += dense_normalized_input

            output = self.relu(output)
            outputs.append(output)

        return outputs[-1]



class DSNet(nn.Module):
    """Defining the whole model. 
    In high level: 
        - Input -> [batch, 3, height, width]
        - Beginning Layer -> [batch, 3, height, width]
        - First Block: n*_DenseBlock(16) -> [batch, 16, height, width]
        - Transition: TransitionBlock(16, 32) -> [batch, 32, height, width]
        - Second Block: n*_DenseBlock(32) -> [batch, 32, height, width]
        - Transition: TransitionBlock(32, 64) -> [batch, 32, height, width]
        - Third Block: n*_DenseBlock(64) -> [batch, 64, height, width]
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # ResNet blocks [16, 32, 64]
        # first block, 16 channels
        self.residual_layers.append(_DenseBlock(16, self.model_n, device).to(device))
        
        # second block, 32 channels
        self.residual_layers.append(_DenseBlock(32, self.model_n, device, stride=2, down=True).to(device))

        # third block, 64 channels
        self.residual_layers.append(_DenseBlock(64, self.model_n, device, stride=2, down=True).to(device))


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
