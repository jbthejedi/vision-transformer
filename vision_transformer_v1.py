import torch
import torchvision
from torch.utils.data import DataLoader

import random
from torch.utils.data import Subset

# import torch.optim as optim
import numpy as np
from torch.utils.data import random_split
from einops import repeat
from torchvision.datasets import OxfordIIITPet, MNIST
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms.functional import to_pil_image

import torchvision.transforms as T
from dataclasses import dataclass


from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor

torch.manual_seed(1337)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels = 3,
        patch_size = 8,
        emb_size = 128
    ):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

        from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads,
                                               dropout=dropout)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class ViT(nn.Module):
    def __init__(
        self,
        ch=3,
        img_size=144,
        patch_size=4,
        emb_dim=32,
        n_layers=6,
        out_dim=37,
        dropout=0.1,
        heads=2
    ):
        super().__init__()

        # Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(
            in_channels=ch,
            patch_size=patch_size,
            emb_size=emb_dim,
        )
        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout = dropout))))
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))


    def forward(self, img):
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:, 0, :])
        
def main():
    img_size = 32
    batch_size = 16
    emb_dim = 64
    n_layers = 2
    patch_size = 4
    n_epochs = 1000
    p_dropout = 0
    
    # img_size = 144
    # emb_dim = 16
    # batch_size = 8
    # n_layers = 6
    # patch_size = 4
    # n_epochs = 100
    # p_dropout = 0.1

    transform_training_data = T.Compose([
        T.RandomCrop(32, padding=4),
        T. Resize((img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='.', train=True, download=True, transform=transform_training_data
    )
    # trainloader = torch.utils.data.DataLoader(
    #     train_data, batch_size=batch_size, shuffle=True, num_workers=2
    # )
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # dataset = OxfordIIITPet(
    #     root=".", 
    #     download=True, 
    #     transform=T.Compose([
    #         T.Resize((img_size, img_size)),
    #         T.ToTensor()
    #     ])
    # )
    # num_samples = 100
    # random_indices = random.sample(range(len(dataset)), num_samples)
    # random_subset = Subset(dataset, random_indices)
    # train_split = int(0.8 * len(random_subset))
    # train, test = random_split(random_subset, [train_split, len(random_subset) - train_split])
    
    train_split = int(0.1 * len(dataset))
    print(f"train size {train_split}")
    # train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    train, test = random_split(dataset, [train_split, 100])
    
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = ViT(
        ch=3,
        img_size=img_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        n_layers=n_layers,
        out_dim=37,
        dropout=p_dropout,
        heads=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        print(epoch)
        epoch_losses = []
        model.train()
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch % 5 == 0:
        # if epoch % 1 == 0:
            print(f">>> Epoch {epoch} train loss: {np.mean(epoch_losses):.4f}")
            epoch_losses = []
            # Something was strange when using this?
            model.eval()
            for step, (inputs, labels) in enumerate(test_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_losses.append(loss.item())
            print(f">>> Epoch {epoch} test loss: {np.mean(epoch_losses):.4f}")
            print()

if __name__ == "__main__":
    main()