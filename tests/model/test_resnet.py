
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import model_abc
from core.model.resnet import ResNet


def test_resnet():
    sample_data = torch.rand(20, 3, 200, 200)
    model = ResNet(model_n=3)

    output = model.forward(sample_data)
    assert output.shape == (20, 10)
