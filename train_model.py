import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_path, 'data', 'datasets', 'Combined Data.csv')

print("Loading dataset...")
df = pd.read_csv(csv_path).dropna(subset=['statement', 'status'])

# -------------------------
# USE LARGE DATA SAMPLE
# -------------------------
if len(df) > 30000:
    df = df.sample(n=30000, random_state=42)

# -------------------------
# BALANCE DATASET
# -------------------------
min_size = df['status'].value_counts().min()

df = (
    df.groupby('status')
    .sample(min_size, random_state=42)
    .reset_index(drop=True)
)

print("Balanced class distribution:")
print(df['status'].value_counts())

# Encode labels
df['diag_label'], classes = pd.factorize(df['status'])
num_classes = len(classes)

df['severity'] = df['status'].apply(lambda x: 0 if x.lower() == "normal" else 1)

# -------------------------
# CLASS WEIGHTS
# -------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df['diag_label']),
    y=df['diag_label']
)

class_weights = torch.tensor(class_weights, dtype=torch.float)

# -------------------------
# DATASET CLASS
# -------------------------
class MentalDataset(Dataset):

    def __init__(self, texts, diag, sev, tokenizer):
        self.texts = texts
        self.diag = diag
        self.sev = sev
        self.tokenizer = tokenizer

    def __getitem__(self, idx):

        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return {
            "ids": enc["input_ids"].squeeze(),
            "mask": enc["attention_mask"].squeeze(),
            "diag": torch.tensor(self.diag[idx]),
            "sev": torch.tensor(self.sev[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.texts)

# -------------------------
# MODEL
# -------------------------
class MentalModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(768, 64, batch_first=True)

        self.severity_head = nn.Linear(64, 1)
        self.diagnosis_head = nn.Linear(64, num_classes)

    def forward(self, ids, mask):

        with torch.no_grad():
            out = self.bert(ids, attention_mask=mask)

        _, (h, _) = self.lstm(out.last_hidden_state)

        h = h[-1]

        severity = torch.sigmoid(self.severity_head(h))
        diagnosis = self.diagnosis_head(h)

        return severity, diagnosis


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = MentalDataset(
    df['statement'].tolist(),
    df['diag_label'].tolist(),
    df['severity'].tolist(),
    tokenizer
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = MentalModel(num_classes)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4
)

loss_diag = nn.CrossEntropyLoss(weight=class_weights)
loss_sev = nn.BCELoss()

epochs = 3

print("Training model...")

model.train()

for epoch in range(epochs):

    total_loss = 0

    for batch in loader:

        optimizer.zero_grad()

        sev_pred, diag_pred = model(batch["ids"], batch["mask"])

        l1 = loss_sev(sev_pred.squeeze(), batch["sev"])
        l2 = loss_diag(diag_pred, batch["diag"])

        loss = l1 + l2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")

save_path = os.path.join(base_path, "ai_engine", "mental_model.pth")

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes.tolist()
}, save_path)

print("Model saved successfully")