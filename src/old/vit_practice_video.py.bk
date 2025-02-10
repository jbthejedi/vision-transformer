import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import transformer_modules as tm

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass
from tqdm import tqdm

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class Config:

    n_epochs        : int    = 10
    p_train_split   : float  = 0.8
    
    image_size      : int    = 32
    batch_size      : int    = 32
    learning_rate : float = 1e-3
    n_embd          : int    = 128
    patch_size      : int    = 4
    n_classes       : int    = 10
    # n_channels      : int    = 3
    num_channels      : int    = 1
    n_heads         : int    = 4
    n_blocks : int    = 2

    # Layer norm parameters
    bias            : bool   = True
    eps             : float  = 1e-5
    momentum        : float  = 0.1

    p_dropout       : float  = 0.1

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        C, p = config.num_channels, config.patch_size
        self.proj = nn.Linear(C*p*p, config.n_embd)

    def forward(self, x):
        """
        in.shape = (b, c, h, w)
        out.shape = (b, n, d)
        b = batch_size
        n = p * p, where p = patch size
        d = number of dimesions in patch embeddings
        """
        b, c_in, h_in, w_in = x.shape
        p = self.config.patch_size
        x = x.unfold(2, p, p) # (b, c, h/p, p, w)
        x = x.unfold(3, p, p) # (b, c, h/p, w/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous() # (b, h/p, w/p, c, p, p)
        x = x.view(b, -1, c_in*p*p) # (b, h/p*w/p, c, p, p)
        x = self.proj(x) # (b, n, d)
        
        return x

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        b = config.batch_size
        n = (config.image_size // config.patch_size) ** 2
        d = config.n_embd
        print("shapes b n d")
        print(b, n, d)

        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.01)
        self.pos_embedding = nn.Parameter(torch.randn(1, n+1, d) * 0.01)

        self.transformer_blocks = nn.Sequential(
            *[tm.TransformerBlock(config) for _ in range(config.n_blocks)]
        )
        self.ln = tm.MyLayerNorm(dim=d)
        self.head = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape

        patches = self.patch_embedding(x) # (b, n, d) where n = h/p * w/p
        cls_tokens = self.cls_token.expand((b, -1, -1)) # (b, 1, d)
        x = torch.cat([cls_tokens, patches], dim=1) # (b, n+1, d)
        x = x + self.pos_embedding[:, :x.size(1), :] # (b, n+1, d)

        x = self.transformer_blocks(x)
        x = self.ln(x[:, 0, :]) # grab only the cls_token
        logits = self.head(x)

        return logits
        
def test_modules(config : Config):
    in_tensor = torch.zeros(8, 3, 32, 32)
    module = PatchEmbedding(config)

    # out should be shape (b, n, d)
    # b = batch_size, n = h/p*w/p, where p = patch_size,
    # d = num dims of embeddings
    out_tensor = module(in_tensor)
    
    expected_shape = (8, 64, 32)
    assert out_tensor.shape == expected_shape, f"Failed. Expected shape {expected_shape} but got {out_tensor.shape}"

    # ---------------
    # ViT
    # ---------------
    # in_tensor = torch.zeros(8, 3, 32, 32)
    # module = PatchEmbedding(config)

    # # out should be shape (b, n, d)
    # # b = batch_size, n = h/p*w/p, where p = patch_size,
    # # d = num dims of embeddings
    # out_tensor = module(in_tensor)
    # 
    # expected_shape = (8, 10)
    # assert out_tensor.shape == expected_shape, f"Failed. Expected shape {expected_shape} but got {out_tensor.shape}"


def train_test_model(config : Config):
    dataset = MNIST(
        root="..",
        download=False,
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Normalize((0.5), (0.5)),
        ])
    )
    train_split = int(config.p_train_split * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    train_dl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    model = ViT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")

        model.train()
        train_loss = 0
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
        
        model.eval()
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
    
    # visualize_predictions(model, test_dl, num_images=10)
    print("end")

def main():
    config = Config
    # test_modules(config)
    train_test_model(config)

if __name__ == '__main__':
    main()
