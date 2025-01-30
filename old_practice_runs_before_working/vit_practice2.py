import torch
import torchvision
import random
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torchvision.transforms as T

from torch import nn
from dataclasses import dataclass, asdict
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pprint import pprint

from tqdm import tqdm

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x):
        # defines the dimensions over which to apply the normalization
        normalized_shape=x.shape[-1:]
        return F.layer_norm(
            x,
            normalized_shape,
            self.weight,
            self.bias,
            eps=1e-3
        )

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cpp = config.n_channels * config.patch_size * config.patch_size
        self.proj = nn.Linear(cpp, config.n_embd)

    def forward(self, x):
        """
        p = patch_size
        C = num_channels
        N = (H/p) * (W/p)
        (B, C, H, W) => (B, N, n_embd)
        """
        p = self.config.patch_size
        B, C, H, W = x.shape
        patches = x.unfold(2, p, p) # (B, C, H/p, p, W)
        patches = patches.unfold(3, p, p) # (B, C, H/p, W/p, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # (B, H/p, W/p, C, p, p)
        patches = patches.view(B, -1, C*p*p) # (B, (H/p)*(W/p), C*p*p)
        out = self.proj(patches) # (B, N, n_embd)
        return out

class AttentionHead(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        B, N, n_embd = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (n_embd**-0.5)
        att = F.softmax(wei, dim=-1)
        att = self.dropout(att)
        out = wei @ v
        
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead(config, head_size) for _ in range(config.n_heads)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        x = torch.cat([att_head(x) for att_head in self.attention_heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.p_dropout),
        )

    def forward(self, x):
        return self.net(x)
        
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
        
class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = PatchEmbedding(config) # (B, N, n_embd)
        self.cls_token = nn.Parameter(torch.zeros((1, 1, config.n_embd)))
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches+1, config.n_embd))
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_blocks)]
        )
        self.ln = LayerNorm(config)
        self.head = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x) # (B, N, n_embd)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, n_embd)
        x = torch.cat([cls_tokens, x], dim=1) # (B, N+1, n_embd)
        
        pos_emb = self.pos_emb[:, :x.size(1), :]
        # print(f"pos_emb.shape {pos_emb.shape}")
        x = x + pos_emb # (B, N+1, n_embd)
        
        x = self.transformer_blocks(x) # (B, N+1, n_embd)
        x = self.ln(x[:, 0, :]) # select class token => (B, n_embd) 
        logits = self.head(x) # (B, n_classes)
        
        return logits
    
@dataclass
class Config:
    n_embd : int = 32
    n_heads : int = 2
    patch_size : int = 4
    n_blocks : int = 1
    n_classes : int = 37
    n_channels : int = 1
    batch_size : int = 16
    n_epochs : int = 3
    
    bias : bool = True
    p_dropout : float = 0.1
    
    image_size : int = 32

def train_test_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE {device}")
    
    config = Config()
    pprint(asdict(config))
    
    dataset = MNIST(
        root=".",
        download=True,
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
    )
    train_split = int(0.8 * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    train_dl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    model = ViT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.n_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        print("Train")
        print(f"Epoch {epoch+1}/{config.n_epochs}")
        for inputs, labels in tqdm(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            print("Validation")
            for inputs, labels in tqdm(test_dl):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        print(f"Train Loss {train_epoch_loss:.4f} Val Loss {val_epoch_loss:.4f}")
        print(f"Train Acc {train_epoch_acc:.2f} Val Acc {val_epoch_acc:.2f}")

def assert_shape(module_name, out_shape, expected_shape):
    assert out_shape == expected_shape, (
        f"{module_name} output shape expected to be {expected_shape}, but got {out_shape}"
    )
    print(f"{module_name} output shape {out_shape} - Passed") 
        
def test_architecture(): 
    # Head test in/out
    config = Config()
    ones = torch.ones((1, 16, 32))
    module = AttentionHead(config, head_size=16)
    out = module(ones)
    expected_shape = (1, 16, 16)
    assert_shape("AttentionHead", out.shape, expected_shape)
    
    # MultiHeadAttention test in/out
    config = Config()
    ones = torch.ones((1, 4, config.n_embd))
    module = MultiHeadAttention(config)
    out = module(ones)
    print(f"MultiHeadAttention out.shape {out.shape}")
    
    # TransformerBlock test in/out
    config = Config()
    ones = torch.ones((1, 4, config.n_embd))
    module = TransformerBlock(config)
    out = module(ones)
    print(f"TransformerBlock out.shape {out.shape}")
    
def main():
    test_architecture()
    train_test_model()
    
if __name__ == '__main__':
    main()























    