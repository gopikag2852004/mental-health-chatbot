import os
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import BertTokenizer
from sklearn.svm import SVC
from tqdm import tqdm

from .model import MentalModel  # your previous MentalModel

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PATHS
# -----------------------------
base_path = Path(__file__).resolve().parent.parent
DATA_PATH = base_path / "data" / "datasets" / "Combined Data.csv"
MODEL_PATH = base_path / "ai_engine" / "mental_model.pth"
SVM_PATH = base_path / "ai_engine" / "svm_model.pkl"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv(DATA_PATH).dropna(subset=['statement', 'status'])
df['statement'] = df['statement'].astype(str).str.strip()

# remove empty strings
df = df[df['statement'] != ""]

texts = df['statement'].tolist()

# dynamic labels
CLASSES = sorted(df['status'].unique())
label_map = {name: i for i, name in enumerate(CLASSES)}
labels = [label_map[l] for l in df['status']]

print("Total samples:", len(texts))
print("Detected classes:", CLASSES)

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -----------------------------
# LOAD MODEL
# -----------------------------
num_classes = len(CLASSES)
model = MentalModel(num_classes=num_classes).to(device)

ckpt = torch.load(MODEL_PATH, map_location=device)
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt

# filter out unexpected keys
model_state = model.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if k in model_state}
model.load_state_dict(filtered_dict, strict=False)
model.eval()

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
BATCH_SIZE = 32
features_list = []
labels_list = []

print("Extracting LSTM features in batches...")

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_labels = labels[i:i+BATCH_SIZE]

    enc = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        _, features = model(enc["input_ids"], enc["attention_mask"])

    features_list.extend(features.cpu().numpy())
    labels_list.extend(batch_labels)

X = np.array(features_list)
y = np.array(labels_list)
print("Feature matrix shape:", X.shape)

# -----------------------------
# TRAIN SVM
# -----------------------------
print("Training SVM classifier...")
svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
svm.fit(X, y)

# -----------------------------
# SAVE SVM
# -----------------------------
joblib.dump(svm, SVM_PATH)
print("✅ SVM training complete")
print("Model saved to:", SVM_PATH)