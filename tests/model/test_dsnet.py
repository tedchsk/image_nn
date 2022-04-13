import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import model_abc
from core.model.dsnet import BasicBlock, DSNet, TransitionBlock


def test_densenet_blocks():
    sample_data = torch.rand(20, 16, 200, 200)
    block = BasicBlock(16, n_models=3)

    output = block.forward(sample_data)
    assert output.shape == (20, 16, 200, 200)

    transition = TransitionBlock(16, 32)

    output = transition(output)  # same as transition.forward(output)
    assert output.shape == (20, 32, 100, 100)


def test_densenet_model():
    sample_data = torch.rand(20, 3, 32, 32)
    model = DSNet(3)

    output = model(sample_data)
    assert output.shape == (20, 10)
