from abc import ABC
import torch.nn as nn


class ModelABC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = None

    def forward(self, x):
        pass

