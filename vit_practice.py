import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt
import random

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from dataclasses import dataclass

seed = 1337
torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device {device}")

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x):
        x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, eps=1e-3)
        return x

class AttentionHead(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        # head_size = config.n_embd // config.n_heads
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        B, T, C = x.shape
        print(f"x.shape {x.shape}")
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        att = F.softmax(wei, dim=-1)
        att = self.dropout(att)

        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.sa_heads = nn.ModuleList([AttentionHead(config, head_size) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.sa_heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.p_dropout),
        )
        
    def forward(self, x):
        return self.ffwd(x)
        
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mha(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        C = config.in_channels * config.patch_size * config.patch_size
        self.proj = nn.Linear(C, config.n_embd)
        

    def forward(self, x):
        """
        x.shape => (B, C, H, W)
        B = batch_size
        C = num_channels
        p = patch_size
        N = (H/p)*(W/p) number of patches
        Returns: (B, N, n_embd)
        """
        B, C, H, W = x.shape
        p = self.config.patch_size
        x = x.unfold(2, p, p) # shape => (B, C, H/p, p, W)
        patches = x.unfold(3, p, p) # shape => (B, C, H/p, W/p, p, p)
        
        # We want to go from (B, C, H/p, W/p, p, p) => (B, H/p, W/p, C, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        # We want to go from (B, H/p, W/p, C, p, p) => (B, N, C*p*p)
        # where N = (H/p) * (W/p) is number of patches (num_patches)
        patches = patches.view(B, -1, C*p*p)
        out = self.proj(patches)
        return out

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.patch_size) ** 2
        
        self.patch_embedding = PatchEmbedding(config)
        
        # cls_token.shape => (B, 1, n_embd)
        self.cls_token = nn.Parameter(torch.zeros((1, 1, config.n_embd)))
        
        # pos_embd.shape => (B, N+1, n_embd), where N = num_patches = (H/p) * (W/p)
        self.pos_emb = nn.Parameter(torch.zeros((1, num_patches+1, config.n_embd)))

        self.t_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_blocks)]
        )
        self.ln = LayerNorm(config)
        self.head = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, out_dim)
        """
        B = x.shape[0]
        x = self.patch_embedding(x) # (B, N, n_embd)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, N, n_embd)
        x = torch.cat([x, cls_token], dim=1) # (B, N+1, n_embd)

        # Flexibility: While in this implementation, num_patches
        # is fixed based on the image size and patch
        # size, using x.size(1) allows for flexibility.
        # If, for some reason, the sequence length changes
        # (e.g., different image sizes or dynamic patching), this
        # indexing ensures that the positional embeddings align
        # correctly with the input sequence.
        pos_emb = self.pos_emb[:, :x.size(1), :] # (B, N+1, n_embd)
        x = x + pos_emb # (B, N, n_embd)
        x = self.t_blocks(x) # (B, N, n_embd)
        x = self.ln(x[:, 0, :]) # (B, n_embd)
        logits = self.head(x)
        return logits
        

@dataclass
class Config:
    n_embd : int       = 32
    n_heads : int      = 4
    n_layers : int     = 1
    patch_size : int   = 4
    n_blocks : int     = 2
    n_classes : int    = 37
    in_channels : int  = 1
    # in_channels : int  = 3
  
    bias : bool        = True
    p_dropout : float  = 0.1

    image_size : int   = 64

def main():
    # print('hello world')
    config = Config

    # # Test LayerNorm
    module = LayerNorm(config)
    # tensor_in = torch.ones((1, 1, 32, 32)) 
    tensor_in = torch.ones((1, 16, 32)) 
    out = module(tensor_in)
    print(f"LayerNorm out.shape {out.shape}")

    # Test Attention
    module = AttentionHead(config, head_size=32)
    tensor_in = torch.ones((1, 16, 32)) 
    out = module(tensor_in)
    print(f"Head out.shape {out.shape}")
    
    # Test MultiHeadAttention
    module = MultiHeadAttention(config)
    tensor_in = torch.ones((1, 16, 32)) 
    out = module(tensor_in)
    print(f"MultiHeadAttention out.shape {out.shape}")
    
    # Test TransformerBlock
    module = TransformerBlock(config)
    tensor_in = torch.ones((1, 16, 32)) 
    out = module(tensor_in)
    print(f"Transformer Block out.shape {out.shape}")

    # Test TransformerBlock
    module = ViT(config)
    tensor_in = torch.ones((1, 1, 32, 32)) 
    out = module(tensor_in)
    print(f"ViT out.shape {out.shape}")
    
    # Test PatchEmbedding
    # module = PatchEmbedding(config)
    # print(f"PatchEmbedding out.shape {out.shape}")
    
    # z = torch.zeros(1, 1, config.n_embd)
    # print(z.shape)

    dataset = MNIST(
        root=".",
        download=True,
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
    )
    # train_split

if __name__ == '__main__':
    main()
















    