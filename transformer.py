import torch
import torch.nn.functional as F
        
from torch.utils.data import IterableDataset, DataLoader

from torch import nn
from pathlib import Path
from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        
        tril = torch.tril(torch.ones(config.cw_size, config.cw_size))
        self.register_buffer('tril', tril)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.sa_heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.sa_heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_heads = MultiHeadAttention(config)
        self.ffwd = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.p_dropout),
        )
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.ln2 = LayerNorm(config.n_embd, config.bias)

    def forward(self, x):
        x = self.ln1(x) 
        x = x + self.sa_heads(x)
        x = self.ln2(x) 
        x = x + self.ffwd(x)
        
        return x

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding_table = nn.Embedding(config.cw_size, config.n_embd)
        self.sa_blocks = nn.Sequential(
            Block(config),
            Block(config),
            Block(config),
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_blocks(x)
        logits = self.lm_head(x)
        
        if targets == None:
            losses = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            losses = F.cross_entropy(logits, targets)
            
        return logits, losses
        
    def generate(self, idx, max_new_tokens=1000):
        idx = idx.to(device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.cw_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
        
class TextDataset(Dataset):
    def __init__(self, data, cw_size):
        self.data = data
        self.cw_size = cw_size
        self.length = len(data) - cw_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.cw_size]
        y = self.data[idx + 1 : idx + self.cw_size + 1]
        return x, y

class Data:
    def encode(self, x):
        return [self.stoi[s] for s in x]
        
    def decode(self, x):
        return [self.itos[s] for s in x]
        
    def __init__(self, config):
        self.vocab = sorted(list(set(config.text)))
        config.vocab_size = len(self.vocab)
        
        self.stoi = { ch: i for i, ch in enumerate(self.vocab) }
        self.itos = { i: ch for i, ch in enumerate(self.vocab) }
        
        data = torch.tensor(self.encode(config.text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        self.train_dataset = TextDataset(train_data, config.cw_size)
        self.val_dataset = TextDataset(val_data, config.cw_size)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            drop_last=True
        )

    @torch.no_grad()
    def estimate_loss(self, model, config):
        out = {}
        model.eval()
        # print("estimate_loss")
        for split in ['train', 'val']:
            loader = self.train_loader if split == 'train' else self.val_loader
            losses = []
            for X, Y in loader:
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
                losses.append(loss.item())
                if len(losses) >= config.eval_iters:
                    break
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

            
@dataclass
class Config:
    max_iters: int = 5000
    eval_iters: int = 200
    text: str = ""
    
    # cw_size: int = 32
    # batch_size: int = 64
    # n_embd: int = 384
    # n_heads: int = 4
    
    cw_size: int = 8
    batch_size: int = 8
    n_embd: int = 32
    n_heads: int = 4
    vocab_size: int = None

    p_dropout : float = 0.2
    bias : bool = True
    
def main():
    config = Config()
    with Path("/Users/justinbarry/projects/language-modeling/input.txt").open("r", encoding="utf-8") as f:
        config.text = f.read()
    data = Data(config)

    # Model
    m = LanguageModel(config)
    m = m.to(device)

    # Train
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for iter_ in range(config.max_iters):
        # print("iter_")
        if iter_ % config.eval_iters == 0:
            out = data.estimate_loss(m, config)
            print(f"Iter {iter_}: train loss {out['train']:.4f} val loss {out['val']:.4f}")
        
        # Iterate over the DataLoader
        for xb, yb in data.train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = m(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            break  # Remove this break to train on the entire epoch
        
        # Optionally, implement additional training logic here

    # Generate
    m.eval()
    out = m.generate(torch.zeros((1, 1), dtype=torch.long).to(device))
    print("".join(data.decode(out[0].tolist())))
        

if __name__ == '__main__':
    main()