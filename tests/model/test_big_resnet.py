import torch
import pytest

from core.model.big_resnet import ResNet18


def test_resnet_18():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    assert y.shape == (1, 100)
