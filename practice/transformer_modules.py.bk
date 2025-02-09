import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)

        tril = torch.tril(torch.ones(config.cw_size, config.cw_size))
        self.register_buffer("tril", tril)

        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        b, t, d = x.shape

        # What information am I looking for?
        q = self.query(x) # (b, t, h), where h=head_size

        # What information do I contain?
        k = self.key(x)  # (b, t, h)

        # What information do I want to
        # communicate during aggregation of token values?
        v = self.value(x)  # (b, t, h)

        att = q @ k.transpose(1, 2) * (d**-0.5) # (b, t, t)
        att = att.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att) # (b, t, t)
        return att @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead(config, head_size) for _ in range(config.n_heads)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        x = self.proj(x)
        return x

class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_mean = x.mean(-1, keepdims=True)
        x_var = x.var(-1, keepdims=True)
        x_hat = (x - x_mean) / torch.sqrt(x + self.eps)
        x_hat = x * self.gamma + self.beta
        return x_hat

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
        self.ln1 = MyLayerNorm(dim=config.n_embd)
        self.ln2 = MyLayerNorm(dim=config.n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mha(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x






