import torch
import torch.nn as nn
from transformers import BertModel


class MentalModel(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # freeze BERT for stability
        for p in self.bert.parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # severity prediction stays neural
        self.severity_head = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask):

        with torch.no_grad():
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        lstm_out, (hidden, _) = self.lstm(bert_out.last_hidden_state)

        features = hidden[-1]     # 64-dim feature vector

        severity = torch.sigmoid(self.severity_head(features))

        return severity, features
