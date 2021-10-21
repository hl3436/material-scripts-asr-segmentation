import torch
import torch.nn as nn
import math


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab, embed_dim, dropout=0.1, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(
            len(vocab), embedding_dim=embed_dim, padding_idx=vocab.pad()
        )

        self.position_embed = nn.Embedding(max_len, embedding_dim=embed_dim)

        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def forward(self, x):
        token_embed = self.token_embed(x)

        position_ids = self.position_ids[:, : x.size(1)]
        position_embed = self.position_embed(position_ids)

        embeddings = token_embed + position_embed

        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
