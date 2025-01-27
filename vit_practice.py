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
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x):
        x = torch.layer_norm(x, x.shape[-1:], self.weight, self.bias, eps=1e-3)
        