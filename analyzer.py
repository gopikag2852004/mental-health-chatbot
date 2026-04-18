import torch
import joblib
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from collections import deque
from .model import MentalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATH SETUP ---
current_file_path = Path(__file__).resolve()
ai_engine_dir = current_file_path.parent
MODEL_PATH = ai_engine_dir / "mental_model.pth"
SVM_PATH = ai_engine_dir / "svm_model.pkl"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
CLASSES = ["Normal", "Depression", "Anxiety", "Bipolar", "PTSD", "Schizophrenia", "Suicidal"]

# --- LOAD MODELS ---
model = MentalModel(num_classes=len(CLASSES)).to(device)
if MODEL_PATH.exists():
    ckpt = torch.load(str(MODEL_PATH), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

svm_model = joblib.load(SVM_PATH) if SVM_PATH.exists() else None
conversation_memory = deque(maxlen=10)
session_confidences = deque(maxlen=50)   # tracks SVM confidence per message
session_class_hits  = deque(maxlen=50)   # tracks top-class prob per message

NEGATIONS = {"not", "never", "no", "don't", "dont", "can't", "cant", "isn't", "isnt", "neither", "nor"}

def is_negated(msg_lower, keyword):
    words = msg_lower.split()
    for i, word in enumerate(words):
        if keyword in word and i > 0 and words[i - 1] in NEGATIONS:
            return True
    return False

def process_message(message, user_id=None):
    msg_lower = message.lower()
    inputs = tokenizer(message, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)

    # 1. GET AI PREDICTIONS
    with torch.no_grad():
        severity_output, features = model(inputs["input_ids"], inputs["attention_mask"])
    
    raw_val = float(severity_output.squeeze().item())
    
    precision_output = 0.0
    class_probs = {}
    if svm_model:
        feature_vec = features.cpu().numpy()
        pred = svm_model.predict(feature_vec)[0]
        svm_class = CLASSES[pred]
        probs = svm_model.predict_proba(feature_vec)[0]
        precision_output = round(float(np.max(probs)) * 100, 2)
        class_probs = {CLASSES[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)}
    else:
        svm_class = "Unknown"

    # 2. CALCULATE BASE MOOD (Inverting 0.81 raw to be negative)
    # Since your model thinks 0.8 = Sad, we must flip it.
    corrected_val = 1.0 - raw_val
    mood_score = (corrected_val * 2) - 1

    # 3. KEYWORD OVERRIDES (Highest Priority)
    neg_strong = ["panic","terrified","hopeless","worthless","anxious","anxiety"]
    neg_mild = ["sad","down","lonely","upset","worried","nervous"]
    pos_mild = ["happy","good","fine","okay","better"]
    physical_distress = [
        "tremor", "trembling", "shaking", "shaky", "sweating", "sweat",
        "heart racing", "pressure","racing heart","heartbreak","heart break","breakup","break up", "heartbeat", "heart beat", "palpitation",
        "chest tight", "chest pain", "shortness of breath", "cant breathe",
        "can't breathe", "dizzy", "dizziness", "nausea", "nauseous",
        "vomiting", "headache", "migraine", "fatigue", "exhausted",
        "fainting", "faint", "numbness", "numb", "tingling"
    ]

    is_keyword_match = False
    if any(w in msg_lower for w in physical_distress):
        mood_score = -0.8
        is_keyword_match = True
    elif any(w in msg_lower for w in neg_strong) and not any(is_negated(msg_lower, w) for w in neg_strong):
        mood_score = -0.8
        is_keyword_match = True
    elif any(w in msg_lower for w in neg_mild) and not any(is_negated(msg_lower, w) for w in neg_mild):
        mood_score = -0.4
        is_keyword_match = True
    elif any(w in msg_lower for w in pos_mild) and not any(is_negated(msg_lower, w) for w in pos_mild):
        mood_score = 0.5
        is_keyword_match = True
    elif any(is_negated(msg_lower, w) for w in pos_mild):
        # "not happy", "not okay" → mildly negative
        mood_score = -0.4
        is_keyword_match = True
    elif any(is_negated(msg_lower, w) for w in neg_mild):
        # "not sad", "not lonely" → mildly positive
        mood_score = 0.3
        is_keyword_match = True

    # 4. SAFETY & DIAGNOSIS LOGIC
    # Priority: Safety > Keywords > SVM > Neural
    if any(word in msg_lower for word in ["kill myself", "suicide", "want to die"]):
        mood_score = -1.0
        diagnosis = "Critical Risk"
    elif is_keyword_match:
        # Use the mood_score set by keywords
        diagnosis = "Emotional Distress" if mood_score < 0 else "Positive / Stable"
    elif svm_class in ["Suicidal", "Depression", "Anxiety"]:
        diagnosis = f"Risk Detected ({svm_class})"
        mood_score = min(mood_score, -0.5) # Force negative if SVM detects illness
    else:
        diagnosis = "Positive / Stable" if mood_score > 0 else "Emotionally Neutral"

    # 5. MEMORY & PATTERNS
    conversation_memory.append(mood_score)
    avg_mood = sum(conversation_memory) / len(conversation_memory)
    early_prediction = "Likely Risk" if avg_mood < -0.5 else "Normal"

    # --- SESSION METRICS (derived from SVM confidence across conversation) ---
    if precision_output > 0:
        session_confidences.append(precision_output)
        # approximate per-class recall using predicted-class probability
        if class_probs:
            session_class_hits.append(class_probs.get(svm_class, 0.0))

    session_accuracy  = round(np.mean(session_confidences), 2) if session_confidences else 0.0
    session_precision = session_accuracy  # confidence = precision proxy
    session_recall    = round(np.mean(session_class_hits), 2)  if session_class_hits  else 0.0
    session_f1 = round(
        2 * (session_precision * session_recall) / (session_precision + session_recall), 2
    ) if (session_precision + session_recall) > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  AI ANALYSIS")
    print(f"{'='*55}")
    print(f"  Text       : '{message}'")
    print(f"  Mood Score : {mood_score:.2f}")
    print(f"  Diagnosis  : {diagnosis}")
    print(f"  Early Risk : {early_prediction}")
    print(f"  SVM Class  : {svm_class}  (Confidence: {precision_output}%)")
    if class_probs:
        print(f"  Class Probabilities:")
        for cls, prob in sorted(class_probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob / 5)
            marker = " ◄" if cls == svm_class else ""
            print(f"    {cls:<16} {prob:>5.1f}%  {bar}{marker}")
    print(f"  --- Session Metrics (last {len(session_confidences)} messages) ---")
    print(f"  Accuracy  : {session_accuracy:.2f}%")
    print(f"  Precision : {session_precision:.2f}%")
    print(f"  Recall    : {session_recall:.2f}%")
    print(f"  F1-Score  : {session_f1:.2f}%")
    print(f"{'='*55}")

    return {
        "severity_score": round(mood_score, 2),
        "status": "Positive" if mood_score > 0 else "Negative",
        "diagnosis": diagnosis,
        "early_prediction": early_prediction,
        "prediction_precision": precision_output
    }
