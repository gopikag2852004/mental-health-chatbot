import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, classification_report

from ai_engine.model import MentalModel

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PATHS
# -----------------------------
base_path = Path(__file__).resolve().parent
DATA_PATH = base_path / "data" / "datasets" / "Combined Data.csv"
MODEL_PATH = base_path / "ai_engine" / "mental_model.pth"

print("\n[1/5] Loading dataset...")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH).dropna(subset=["statement", "status"])
df["statement"] = df["statement"].astype(str).str.strip()

# OPTIONAL: speed up
df = df.sample(n=10000, random_state=42)

texts = df["statement"].tolist()

CLASSES = sorted(df["status"].unique())
label_map = {name: i for i, name in enumerate(CLASSES)}
labels = np.array([label_map[l] for l in df["status"]])

print(f"Total samples: {len(texts)}")
print("Classes:", CLASSES)

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -----------------------------
# LOAD MODEL
# -----------------------------
print("\n[2/5] Loading trained model...")

model = MentalModel(num_classes=len(CLASSES)).to(device)

ckpt = torch.load(MODEL_PATH, map_location=device)
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt

model.load_state_dict(state_dict, strict=False)
model.eval()

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
print("\n[3/5] Extracting features (this may take time)...")

BATCH_SIZE = 32
features_list = []

total_batches = len(texts) // BATCH_SIZE + 1

for i in tqdm(range(0, len(texts), BATCH_SIZE),
              total=total_batches,
              desc="Extracting Features"):

    batch = texts[i:i+BATCH_SIZE]

    enc = tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        _, features = model(enc["input_ids"], enc["attention_mask"])

    features_list.extend(features.cpu().numpy())

X = np.array(features_list)
y = labels

print("Feature shape:", X.shape)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
print("\n[4/5] Splitting data and training SVM...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
svm.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)

# -----------------------------
# CONFUSION MATRIX (FIXED)
# -----------------------------
print("\n[5/5] Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[c[:6] for c in CLASSES]  # short labels
)

disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# -----------------------------
# ROC CURVE (SIMPLIFIED)
# -----------------------------
print("Generating simplified ROC curve...")

# Binary: Normal vs Others
normal_index = label_map.get("Normal", label_map.get("normal"))

y_test_binary = (y_test != normal_index).astype(int)
y_prob_binary = 1 - y_prob[:, normal_index]

fpr, tpr, _ = roc_curve(y_test_binary, y_prob_binary)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Normal vs Mental Risk)")
plt.legend()

plt.grid(alpha=0.3)

plt.savefig("roc_curve.png", dpi=300)
plt.show()

# -----------------------------
# REPORT
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASSES))

print("\n✅ DONE — Files saved:")
print("   - confusion_matrix.png")
print("   - roc_curve.png")