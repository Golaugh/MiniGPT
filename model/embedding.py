import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.token_embed(x)