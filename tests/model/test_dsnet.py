import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.dsnet import DSNet


def test_densenet_model():
    sample_data = torch.rand(20, 3, 32, 32)
    model = DSNet(3)

    output = model(sample_data)
    assert output.shape == (20, 10)
