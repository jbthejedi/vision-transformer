import traceback

import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
from dataclasses import dataclass

from tqdm import tqdm

seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE {device}")
print(f"Seed {seed}")

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gamma = nn.Parameter(torch.ones(config.n_embd))
        self.beta = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.config.eps)
        out = self.gamma * x_hat + self.beta
        
        return out

class AttentionHead(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        q = self.query(x) # (B, N, head_size)
        k = self.key(x) # (B, N, head_size)
        v = self.value(x) # (B, N, head_size)
        
        # should this be scalled by sqrt(head_size) and not sqrt(n_embd)?
        att = q @ k.transpose(-2, -1) * (self.head_size**(-0.5)) # (B, N, N)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        
        return out

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
        out = self.net(x)
        return out

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
        out = x + self.ffwd(x)
        
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.mha = nn.ModuleList(
            [AttentionHead(config, head_size) for _ in range(config.n_heads)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.mha], dim=-1)
        x = self.proj(x)
        out = self.dropout(x)
        return out
        
class SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        linear_in_shape = config.image_size * config.image_size * config.n_channels
        self.head = nn.Linear(linear_in_shape, config.n_classes)

    def forward(self, x):
        # print(f"forward x.shape {x.shape}")
        B, C, W, H = x.shape
        x = x.view(B, C*W*H)
        # print(f"after view x.shape {x.shape}")
        logits = self.head(x)
        return logits

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cpp = config.n_channels * (config.patch_size**2)
        self.proj = nn.Linear(cpp, config.n_embd)

    def forward(self, x):
        """
        p = patch_size
        N = (H/p) * (W/p)
        input shape (B, C, H, W)
        output shape (B, N, C*p*p)
        """
        B, C, H, W = x.shape
        p = self.config.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p) # (B, C, H/p, W/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous() # (B, H/p, W/p, C, p, p)
        x = x.view(B, -1, C*p*p)
        x = self.proj(x)
        
        return x
        
class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_emb = PatchEmbedding(config)
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches+1, config.n_embd) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)
        self.t_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln = LayerNorm(config)
        self.head = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, x):
        B, C, W, H = x.shape
        
        # Transform image to patch embeddings with position encodings
        patches = self.patch_emb(x) # (B, N, n_embd)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1) # (B, N+1, n_embd)
        x = x + self.pos_emb[:, :x.size(1), :] # (B, N+1, n_embd)
        x = self.t_blocks(x) # (B, N+1, n_embd)
        x = self.ln(x[:, 0, :])
        logits = self.head(x)
        return logits
        
@dataclass
class Config:
    n_epochs        : int    = 10
    p_train_split   : float  = 0.8
    
    image_size      : int    = 32
    batch_size      : int    = 32
    n_embd          : int    = 128
    patch_size      : int    = 4
    n_classes       : int    = 10
    n_channels      : int    = 3
    n_heads         : int    = 4
    n_layers        : int    = 1

    # Layer norm parameters
    bias            : bool   = True
    eps             : float  = 1e-5
    momentum        : float  = 0.1

    p_dropout       : float  = 0.1

def assert_shape(module_name, out_shape, expected_shape):
    assert out_shape == expected_shape, (
        f"""
        {module_name} output shape expected to be {expected_shape}, but got
        {out_shape}
        """
    )
    print(f"{module_name} output shape {out_shape} - Passed")
    
def test_architecture(config : Config):
    # ----------------
    # Simple MLP
    # ----------------
    print("Test SimpleMLP linear head")
    module = SimpleMLP(config)
    batch_size = 7
    in_tensor = torch.ones((batch_size, config.n_channels, config.image_size, config.image_size))
    print(f"Input shape {in_tensor.shape}")
    out = module(in_tensor)
    expected_shape = (batch_size, config.n_classes)
    assert_shape("SimpleMLP linear head", out.shape, expected_shape)
    print()

    # ----------------
    # PatchEmbedding
    # ----------------
    print("Test PatchEmbedding")
    module = PatchEmbedding(config)
    B = 7
    in_tensor = torch.ones(
        (B, config.n_channels, config.image_size, config.image_size)
    )
    print(f"Input shape {in_tensor.shape}")
    try:
        out = module(in_tensor)
        N = (config.image_size // config.patch_size) ** 2
        expected_shape = (B, N, config.n_embd)
        assert_shape("PatchEmbedding", out.shape, expected_shape)
    except Exception as e:
        print("Test PatchEmbedding Failed with exception")
    print()
        
    # ----------------
    # AttentionHead
    # ----------------
    module_name = "AttentionHead"
    print(f"Test {module_name}")
    B, N, n_embd = 4, 16, config.n_embd
    module = AttentionHead(config, head_size=config.n_embd)
    in_tensor = torch.ones(B, N, n_embd)
    print(f"Input shape {in_tensor.shape}")
    try:
        out = module(in_tensor)
        expected_shape = (B, N, n_embd)
        assert_shape(module_name, out.shape, expected_shape)
    except Exception as e:
        print(f"Test {module_name} Failed with exception {e}")
    print()
    
    # ----------------
    # MultiHeadAttention
    # ----------------
    module_name = "MultiHeadAttention"
    print(f"Test {module_name}")
    B, N, n_embd = 4, 16, config.n_embd
    in_tensor = torch.ones(B, N, n_embd)
    module = MultiHeadAttention(config)
    print(f"Input shape {in_tensor.shape}")
    try:
        out = module(in_tensor)
        expected_shape = (B, N, n_embd)
        assert_shape(module_name, out.shape, expected_shape)
    except Exception as e:
        print(f"Test {module_name} Failed with exception {e}")
    print()
    
    # ----------------
    # LayerNorm
    # ----------------
    module_name = "LayerNorm"
    print(f"Test {module_name}")
    B, N, n_embd = 4, 16, config.n_embd
    in_tensor = torch.ones(B, N, n_embd)
    module = LayerNorm(config)
    print(f"Input shape {in_tensor.shape}")
    try:
        out = module(in_tensor)
        expected_shape = (B, N, n_embd)
        assert_shape(module_name, out.shape, expected_shape)
    except Exception as e:
        print(f"Test {module_name} Failed with exception {e}")
        traceback.print_exc()
    print()
    
    # ----------------
    # TransformerBlock 
    # ----------------
    module_name = "TransformerBlock"
    print(f"Test {module_name}")
    B, N, n_embd = 4, 16, config.n_embd
    in_tensor = torch.ones(B, N, n_embd)
    module = TransformerBlock(config)
    print(f"Input shape {in_tensor.shape}")
    try:
        out = module(in_tensor)
        expected_shape = (B, N, n_embd)
        assert_shape(module_name, out.shape, expected_shape)
    except Exception as e:
        print(f"Test {module_name} Failed with exception {e}")
        traceback.print_exc()
    print()
    
    # ----------------
    # ViT
    # ----------------
    module_name = "ViT"
    print(f"Test {module_name}")
    B, C, W, H = (8, 1, config.image_size, config.image_size) # (8, 1, 32, 32)
    in_tensor = torch.ones(B, C, W, H)
    module = ViT(config)
    print(f"Input shape {in_tensor.shape}")
    try:
        out = module(in_tensor)
        expected_shape = (B, config.n_classes) # (8, 37)
        assert_shape(module_name, out.shape, expected_shape)
    except Exception as e:
        print(f"Test {module_name} Failed with exception {e}")
        traceback.print_exc()
    print()

def visualize_predictions(model, dataloader, num_images=0):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 5))
    for inputs, labels in dataloader:
        logits = model(inputs)
        _, preds = torch.max(logits, 1)

        for i in range(inputs.size(0)): # iterate over batch dimension
            if images_shown >= num_images:
                break
            img = inputs[i].cpu().squeeze()

            true_label = labels[i].item()
            pred_label = preds[i].item()

            plt.subplot(2, num_images//2, images_shown+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"True {true_label} Pred {pred_label}")
            plt.axis('off')

            images_shown += 1
        if images_shown >= num_images:
            break
    plt.tight_layout()
    plt.show()

def train_test_model(config: Config):
    # Dataset and DataLoader setup remains unchanged
    norm_values = (0.5, 0.5, 0.5)
    dataset = CIFAR10(
        root=".",
        download=True,
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.RandomHorizontalFlip(),
            T.RandomCrop(config.image_size, padding=4),
            T.ToTensor(),
            T.Normalize(norm_values, norm_values),
        ]),
    )

    train_split = int(config.p_train_split * len(dataset))
    train, test = random_split(
        dataset, [train_split, len(dataset) - train_split]
    )
    train_dl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    model = ViT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
        
        # ----------------
        # Training Phase
        # ----------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        with tqdm(train_dl, desc="Training", unit="batch", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Optionally update progress bar with current loss
                pbar.set_postfix(loss=loss.item())
        
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        
        # ----------------
        # Validation Phase
        # ----------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with tqdm(test_dl, desc="Validation", unit="batch", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Optionally update progress bar with current loss
                pbar.set_postfix(loss=loss.item())
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # ----------------
        # Logging Epoch Metrics
        # ----------------
        tqdm.write(f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.2f}")
        tqdm.write(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}\n")
    
    visualize_predictions(model, test_dl, num_images=10)


def main():
    config = Config()
    # test_architecture(config)
    train_test_model(config)

if __name__ == '__main__':
    main()
















    