import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        wei = (q @ k.transpose(-2, -1)) / C**0.5
        wei = wei.masked_fill(mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        head_size = embed_dim // n_heads  # so embed_dim should be dividable by n_heads
        self.heads = nn.ModuleList([Head(embed_dim, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, block_size):
        super().__init__()
        self.attn = MultiHead(embed_dim, n_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x