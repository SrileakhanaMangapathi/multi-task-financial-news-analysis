# model_bilstm.py
# BiLSTM multi-task: Market Event + Sentiment with task-specific attention

import torch
import torch.nn as nn

class BiLSTMMultiTask(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 num_event_classes, num_sentiment_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True,
                            bidirectional=True, dropout=dropout, num_layers=2)
        H = hidden_size * 2
        self.event_attn = nn.Linear(H, 1)
        self.sentiment_attn = nn.Linear(H, 1)
        self.event_head = nn.Linear(H, num_event_classes)
        self.sentiment_head = nn.Linear(H, num_sentiment_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        e_ctx = (torch.softmax(self.event_attn(out), 1) * out).sum(1)
        s_ctx = (torch.softmax(self.sentiment_attn(out), 1) * out).sum(1)
        return self.event_head(self.dropout(e_ctx)), \
               self.sentiment_head(self.dropout(s_ctx))
