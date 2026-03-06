# model_bigru.py
# BiGRU + FastText + Attention for Impact Level prediction

import torch
import torch.nn as nn

class BiGRUImpact(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 num_classes, num_numerical=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True,
                          bidirectional=True, dropout=dropout, num_layers=2)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Linear(hidden_size * 2 + num_numerical, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, numerical_features):
        emb = self.dropout(self.embedding(x))
        out, _ = self.gru(emb)                          # (B, T, H*2)
        attn = torch.softmax(self.attention(out), dim=1)
        context = (attn * out).sum(dim=1)               # (B, H*2)
        fused = torch.cat([context, numerical_features], dim=1)
        return self.classifier(self.dropout(fused))
