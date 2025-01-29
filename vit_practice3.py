import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from dataclasses import dataclass

from tqdm import tqdm

seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE {device}")
print(f"Seed {seed}")

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

class ViT(nn.Module):
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
        
@dataclass
class Config:
    image_size      : int    = 32
    n_epochs        : int    = 3
    batch_size      : int    = 32
    train_split     : float  = 0.8
    n_embd          : int    = 32
    n_classes       : int    = 37
    n_show_steps    : int    = 1000
    n_channels      : int    = 1

def assert_shape(module_name, out_shape, expected_shape):
    assert out_shape == expected_shape, (
        f"""
        {module_name} output shape expected to be {expected_shape}, but got
        {out_shape}
        """
    )
    print(f"{module_name} output shape {out_shape} - Passed")
    
def test_architecture(config : Config):
    print("Test init SimpleMLP linear head")
    module = SimpleMLP(config)
    batch_size = 7
    in_tensor = torch.ones((batch_size, config.n_channels, config.image_size, config.image_size))
    out = module(in_tensor)
    expected_shape = (batch_size, config.n_classes)
    assert_shape("SimpleMLP linear head", out.shape, expected_shape)

def train_test_model(config : Config):
    dataset = MNIST(
        root=".",
        download=True,
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            
            # Normalize pixels by mean=(0.5,) std=(0.5,)
            # For training stability and efficiency
            # No vanishing or exploding gradients. No
            # unnecessarily large weights
            T.Normalize((0.5,), (0.5,)),
        ]),
    )

    train_split = int(config.train_split * len(dataset))
    train, test = random_split(
        dataset, [train_split, len(dataset) - train_split]
    )
    train_dl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    model = SimpleMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config.n_epochs):
        print(f"Epoch {epoch}/{config.n_epochs}")
        print("Model Train")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in tqdm(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            # print(f"inputs.shape {inputs.shape}")
            optimizer.zero_grad()
            logits = model(inputs)
            # print(f"logits.shape {logits.shape}")
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        
        print("Model Eval")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_dl):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        print(f"Train Loss {train_epoch_loss:.4f} Val Loss {val_epoch_loss:.4f}")
        print(f"Train Acc {train_epoch_acc:.2f} Val Acc {val_epoch_acc:.2f}")
        

def main():
    config = Config()
    test_architecture(config)
    train_test_model(config)

if __name__ == '__main__':
    main()
















    