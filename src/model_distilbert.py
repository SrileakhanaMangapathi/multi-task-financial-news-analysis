# model_distilbert.py
# DistilBERT fine-tuned for multi-task: Market Event + Sentiment

import torch
import torch.nn as nn
from transformers import DistilBertModel

class DistilBertMultiTask(nn.Module):
    def __init__(self, num_event_classes, num_sentiment_classes, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        hidden = self.bert.config.hidden_size  # 768

        self.event_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_event_classes)
        )
        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_sentiment_classes)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.event_head(cls), self.sentiment_head(cls)
