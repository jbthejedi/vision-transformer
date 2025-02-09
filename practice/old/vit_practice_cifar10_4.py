import torch
import traceback
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torch import nn
from tqdm import tqdm
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'gpu'
seed = 1337
torch.manual_seed(seed)
print(f"DEVICE {device}")
print(f"SEED {seed}")

@dataclass
class Config:
    n_embd         : int = 32 # D
    n_heads        : int = 2
    n_layers       : int = 1
    n_channels     : int = 3
    batch_size     : int = 8
    patch_size     : int = 4
    image_size     : int = 32
    norm_shape     : tuple = (0.5, 0.5, 0.5)

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.n_channels * config.patch_size**2, config.n_embd)

    def forward(self, x):
        """
        x.shape = (B, C, H, W)
        intermediate shape (B, N, C*p*p)
        out.shape = (B, N, n_embd), where N = (H/p) * (W/p)
        """
        B, C, H, W = x.shape
        p = self.config.patch_size 
        x = x.unfold(2, p, p) # (B, C, H/p, p, W)
        x = x.unfold(3, p, p) # (B, C, H/p, W/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C*p*p) # (B, (H/p)*(W/p), C*p*p)
        out = self.proj(x) # (B, N, n_embd)
        return out

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embedding = PatchEmbedding(config)
        self.pos_emb = nn.Parameter(    # (1, P+1, n_embd)
            torch.randn(1, num_patches+1, config.n_embd) * 0.01
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        patch_emb = self.patch_embedding(x) # (B, P, n_emb)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, patch_emb], dim=1) # (B, P+1, n_emb)
        # x.size(1) if x.size(1) < P+1
        x = x + self.pos_emb[:, :x.size(1), :] # (B, P+1, n_emb)
        out = x
        print(f"out.shape {out.shape}")
        return out
        
    
def assert_shape(out_shape, expected_shape):
    assert out_shape == expected_shape, (
        f"""
        Failed
        Expected output.shape {expected_shape}
        Received output.shape {out_shape}
        """
    )
    print(f"Passed: output shape {out_shape} - Passed")
    
def test_architecture(config : Config):
    module_name = "PatchEmbedding"
    module = PatchEmbedding(config)
    in_tensor = torch.ones((8, 3, 32, 32))
    out_tensor = module(in_tensor)
    try:
        assert_shape(out_tensor.shape, expected_shape=(8, 64, 32))
    except Exception as e:
        traceback.print_exc()
    print(f"Test {module_name}")
    print()
    
    module_name = "Vit"
    print(f"Test {module_name}")
    module = ViT(config)
    in_tensor = torch.ones((8, 3, 32, 32))
    out_tensor = module(in_tensor)
    try:
        assert_shape(out_tensor.shape, expected_shape=(8, 65, 32))
    except Exception as e:
        traceback.print_exc()
    print(f"Test {module_name}")
    print()

def train_test_model(config : Config):
    data = CIFAR10(
        root=".",
        download=False,
        transform=T.Compose(
            T.Resize(),
            T.Crop(),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(config.norm_shape, config.norm_shape),
        ),
    )

def main():
    config = Config()
    test_architecture(config)
    
if __name__ == '__main__':
    main()