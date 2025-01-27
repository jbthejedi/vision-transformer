import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import numpy as np
import random
import timeit

from tqdm import tqdm
import matplotlib.pyplot as plt

RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNELS = 1
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS  # 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 49

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# PatchEmbedding module (unchanged)
# -------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x

# -------------------------------------------------------------------
# ViT model (unchanged)
# -------------------------------------------------------------------
class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim,
                 num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        # Take the first (CLS) token only
        x = self.mlp_head(x[:, 0, :])
        return x


# -------------------------------------------------------------------
# Create MNIST datasets using torchvision
# -------------------------------------------------------------------
# Transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the MNIST train set
full_train_dataset = MNIST(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)

# Download the MNIST test set
test_dataset = MNIST(
    root="data",
    train=False,
    download=True,
    transform=val_test_transform
)

# Optionally split the train set into train and val
val_size = 6000   # 10% of 60,000
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Replace transforms for the val dataset (no data augmentation)
# The random_split keeps transform references, but we want to ensure
# the same transform as test (no random rotation):
val_dataset.dataset.transform = val_test_transform

# Create DataLoaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------------------------------------------------
# Initialize model, loss, and optimizer
# -------------------------------------------------------------------
model = ViT(
    num_patches=NUM_PATCHES,
    img_size=IMG_SIZE,
    num_classes=NUM_CLASSES,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_encoders=NUM_ENCODERS,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT,
    activation=ACTIVATION,
    in_channels=IN_CHANNELS
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    betas=ADAM_BETAS,
    lr=LEARNING_RATE,
    weight_decay=ADAM_WEIGHT_DECAY
)

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
start = timeit.default_timer()
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    # --------------------------
    # Train
    # --------------------------
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0

    for idx, (img, label) in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img.to(device)
        label = label.to(device)

        y_pred = model(img)
        loss = criterion(y_pred, label)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

        # Collect predictions
        y_pred_label = torch.argmax(y_pred, dim=1)
        train_labels.extend(label.cpu().numpy())
        train_preds.extend(y_pred_label.cpu().numpy())

    train_loss = train_running_loss / (idx + 1)
    train_accuracy = np.mean(np.array(train_preds) == np.array(train_labels))

    # --------------------------
    # Validation
    # --------------------------
    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0

    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img.to(device)
            label = label.to(device)

            y_pred = model(img)
            loss = criterion(y_pred, label)
            val_running_loss += loss.item()

            y_pred_label = torch.argmax(y_pred, dim=1)
            val_labels.extend(label.cpu().numpy())
            val_preds.extend(y_pred_label.cpu().numpy())

    val_loss = val_running_loss / (idx + 1)
    val_accuracy = np.mean(np.array(val_preds) == np.array(val_labels))

    print("-" * 30)
    print(f"Epoch {epoch+1}:")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss:   {val_loss:.4f},   Val Accuracy:   {val_accuracy:.4f}")
    print("-" * 30)

stop = timeit.default_timer()
print(f"Training Time: {stop - start:.2f}s")

# -------------------------------------------------------------------
# Testing / Inference loop
# -------------------------------------------------------------------
test_labels = []
test_preds = []
test_images = []

model.eval()
with torch.no_grad():
    for idx, (img, label) in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        img = img.to(device)

        outputs = model(img)
        predicted = torch.argmax(outputs, dim=1)

        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(label.cpu().numpy())  # True labels for test set
        test_images.extend(img.cpu().numpy())

# -------------------------------------------------------------------
# Quick visualization of predictions
# -------------------------------------------------------------------
plt.figure()
f, axarr = plt.subplots(2, 3, figsize=(8, 5))
counter = 0
for i in range(2):
    for j in range(3):
        # test_images[counter] has shape (1, 28, 28) because in_channels=1
        axarr[i][j].imshow(test_images[counter].squeeze(), cmap="gray")
        axarr[i][j].set_title(f"Predicted {test_preds[counter]}, True {test_labels[counter]}")
        axarr[i][j].axis("off")
        counter += 1

plt.tight_layout()
plt.show()
