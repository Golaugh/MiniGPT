import torch.nn as nn
from model.embedding import TokenEmbedding
from model.transformer_block import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, n_heads, ff_dim, n_layers):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, ff_dim, block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        tok_emb = self.token_embedding(x)
        x = self.blocks(tok_emb)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits