import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as T
from dataclasses import dataclass

from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class VitConfig:
    # image_size: int = 32      # Image height/width
    image_size: int = 64      # Image height/width
    patch_size: int = 4       # Size of each patch
    # in_channels: int = 3    # Number of channels in input
    in_channels: int = 1      # Number of channels in input
    n_embd: int = 32          # Embedding dimension
    n_heads: int = 2          # Number of attention heads
    n_layers: int = 1         # Number of transformer blocks
    p_dropout: float = 0.1    # Dropout probability
    # out_dim: int = 37       # Number of classes for classification
    out_dim: int = 10         # Number of classes for classification
    bias: bool = True         # Whether to include bias in LayerNorm
    
    n_epochs : int = 3
    batch_size : int = 32
    # batch_size = 8
    emb_dim : int = 16
    # emb_dim : int = 32

# ------------------------------------------------------------------------------
# A simple LayerNorm, matching the style from the language model code
# ------------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)

# ------------------------------------------------------------------------------
# Single-head attention for Vision Transformer (no causal mask)
# ------------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.query = nn.Linear(config.n_embd, head_size)
        self.key   = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        # x shape: (B, T, C)
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention weights
        # q @ k^T -> (B, T, head_size) @ (B, head_size, T) => (B, T, T)
        # We scale by sqrt(head_size) to stabilize gradients
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        # For an encoder, there's no causal mask, so no masking here.
        att = F.softmax(wei, dim=-1)
        att = self.dropout(att)

        # Weighted sum over value vectors
        out = att @ v  # (B, T, head_size)
        return out

# ------------------------------------------------------------------------------
# Multi-head attention: combines multiple Heads, then a linear projection
# ------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        # Concatenate the outputs of each head along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Final linear projection
        out = self.proj(out)
        out = self.dropout(out)
        return out

# ------------------------------------------------------------------------------
# Feed-forward block, same style as the language transformer
# ------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.p_dropout)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# A single transformer block: LN -> MHA -> residual, LN -> FF -> residual
# ------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.sa = MultiHeadAttention(config)  # self-attention
        self.ffwd = FeedForward(config)

    def forward(self, x):
        # x shape: (B, T, C)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ------------------------------------------------------------------------------
# Patch Embedding (like token embedding for images)
# ------------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # We'll map each patch (patch_size*patch_size*in_channels) to n_embd
        self.proj = nn.Linear(config.patch_size*config.patch_size*config.in_channels, config.n_embd)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, N, n_embd), where N = number of patches
        """
        B, C, H, W = x.shape
        p = self.config.patch_size
        # Rearrange into patches: (B, n_patches, patch_area * C)
        # n_patches = (H/p)*(W/p); patch_area = p*p
        # Flatten each patch
        patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        patches = patches.permute(0,2,3,1,4,5).contiguous()
        # shape: (B, H/p, W/p, C, p, p)
        # Flatten the last three dims
        patches = patches.view(B, -1, C*p*p)  # (B, N, patch_area*C)
        # Apply a linear projection
        out = self.proj(patches)  # (B, N, n_embd)
        return out

# ------------------------------------------------------------------------------
# The main Vision Transformer, in a style similar to the decoder language model
# but used as an encoder for images.
# ------------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.config = config

        # 1) Patch + embedding
        self.patch_embedding = PatchEmbedding(config)

        # 2) CLS token & positional embedding
        num_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, config.n_embd))

        # 3) Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])

        # 4) Final LayerNorm, then classification head
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.head = nn.Linear(config.n_embd, config.out_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, out_dim)
        """
        B = x.shape[0]

        # Patch embeddings
        x = self.patch_embedding(x) # (B, N, n_embd)
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, n_embd)
        x = torch.cat((cls_tokens, x), dim=1)         # (B, N+1, n_embd)

        # Add positional embeddings
        x = x + self.pos_emb[:, : x.size(1), :]

        # Pass through Transformer blocks
        x = self.blocks(x)

        # Take the CLS token's representation
        x = self.ln_f(x[:, 0, :])  # (B, n_embd)

        # Classify
        logits = self.head(x)      # (B, out_dim)
        return logits
        
def visualize_predictions(model, dataloader, device, num_images=10):
    """
    Visualize model predictions on a few samples from the dataloader.
    
    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (str): Device to perform computations on.
        num_images (int): Number of images to display.
    """
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 5))
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break
                img = inputs[i].cpu().squeeze()  # Remove channel dimension for grayscale
                true_label = labels[i].item()
                pred_label = preds[i].item()
                
                plt.subplot(2, num_images//2, images_shown+1)
                plt.imshow(img, cmap='gray')
                plt.title(f"True: {true_label}, Pred: {pred_label}")
                plt.axis('off')
                
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.tight_layout()
    plt.show()

def main():

    # Set in_channels to 1 for MNIST and adjust out_dim to 10
    config = VitConfig()  
    dataset = MNIST(
        root=".", 
        download=True, 
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
    )
    train_split = int(0.8 * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    
    train_dataloader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    model = VisionTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config.n_epochs):
        # -----------------------
        # Training Phase
        # -----------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs.shape = (B, ) (32, 1, 32, 32)
            # print(f"inputs.shape {inputs.shape}")
            # exit()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        
        # -----------------------
        # Validation Phase
        # -----------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Accumulate loss
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # -----------------------
        # Logging
        # -----------------------
        print(f"Epoch [{epoch+1}/{config.n_epochs}]")
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_acc*100:.2f}%")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc*100:.2f}%\n")
    
    # -----------------------
    # Visualization After Training
    # -----------------------
    visualize_predictions(model, test_dataloader, device, num_images=10)
    
# Example usage:
if __name__ == "__main__":
    # # Create a config
    # config = VitConfig(image_size=128, patch_size=16, in_channels=3, n_embd=128, n_heads=4, n_layers=4, out_dim=10)

    # # Create the model
    # model = VisionTransformer(config).to(device)

    # # Example input
    # dummy_img = torch.randn(2, 3, 128, 128).to(device)  # batch_size=2, channels=3, 128x128
    # out = model(dummy_img)
    # print("logits shape:", out.shape)  # should be (2, out_dim)

    main()
