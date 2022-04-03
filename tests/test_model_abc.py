import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import model_abc


def test_model_abc():
    class Model(model_abc.ModelABC):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    model = Model()
    output = model.forward(torch.rand(20, 1, 200, 200))
    assert output.shape == (20, 20, 192, 192)
