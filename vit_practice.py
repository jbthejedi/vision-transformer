import torch
import torchvision
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt
import random

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(ndim=config.n_embd)
        self.bias = nn.Parameter(ndim=config.n_embd)

    def forward(self, x):
        x = torch.layer_norm(x, self.weight.shape, self.weight, self.bias, lr=1e-3)