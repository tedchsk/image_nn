import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.densenet import DenseNet
from core.model.resnet import ResNet


def test_densenet_model():
    sample_data = torch.rand(20, 3, 32, 32)
    model = DenseNet(model_n=3, num_classes=10)

    output = model(sample_data)
    assert output.shape == (20, 10)
