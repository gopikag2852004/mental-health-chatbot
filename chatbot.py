# ai_engine/chatbot.py
import json
import random
import os
import torch
from sentence_transformers import SentenceTransformer, util
from collections import deque
from .recommendation_engine import recommend_support
from .risk_detection import RiskDetector

class ChatBotBrain:

    def __init__(self):
        # Load semantic similarity model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load intents
        intents_path = os.path.join('data', 'datasets', 'intents.json')
        with open(intents_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['intents']

        # Collect all patterns
        self.all_patterns = []
        self.pattern_to_intent = []
        for intent in self.data:
            for pattern in intent['patterns']:
                self.all_patterns.append(pattern)
                self.pattern_to_intent.append(intent)

        # Precompute embeddings
        self.pattern_embeddings = self.model.encode(self.all_patterns, convert_to_tensor=True)

        # Risk detector
        self.risk_detector = RiskDetector()

        # Track last 10 messages for early mental illness detection
        self.recent_moods = deque(maxlen=10)


    def detect_mood(self, text):
        text = text.lower()
        if any(word in text for word in ["stress", "stressed", "overwhelmed", "pressure"]):
            return "stress"
        if any(word in text for word in ["anxious", "anxiety", "panic", "nervous"]):
            return "anxiety"
        if any(word in text for word in ["sad", "depressed", "hopeless", "unhappy"]):
            return "sad"
        if any(word in text for word in ["tired", "low", "exhausted"]):
            return "low"
        return "neutral"


    def get_reply(self, user_text):
        # Encode user message
        user_embedding = self.model.encode(user_text, convert_to_tensor=True)

        # Compare with dataset patterns
        cos_scores = util.cos_sim(user_embedding, self.pattern_embeddings)[0]

        # Best match
        best_match_idx = torch.argmax(cos_scores).item()
        max_score = cos_scores[best_match_idx].item()

        if max_score > 0.4:
            matched_intent = self.pattern_to_intent[best_match_idx]
            reply = random.choice(matched_intent['responses'])
        else:
            reply = "I hear you. Could you tell me more about what you're experiencing?"

        # Detect mood
        mood = self.detect_mood(user_text)
        self.recent_moods.append(mood)
        self.risk_detector.add_mood(mood)

        # Generate recommendation only if negative mood
        recommendation = ""
        if mood in ["stress", "anxiety", "sad", "low"]:
            recommendation = recommend_support(mood)

        # Early mental illness detection
        early_warning = ""
        negative_count = sum(1 for m in self.recent_moods if m in ["stress", "anxiety", "sad", "low"])
        if len(self.recent_moods) >= 5 and negative_count >= 3:
            early_warning = "\n⚠️ Multiple concerning messages detected. Consider consulting a mental health professional."

        # Check risk
        risk_flag = self.risk_detector.detect_risk()
        if risk_flag:
            early_warning += "\n⚠️ Repeated negative emotions noticed. Speaking to a professional is advised."

        # Combine reply safely
        full_reply = str(reply)
        if recommendation:
            full_reply += "\n\n" + str(recommendation)
        if early_warning:
            full_reply += str(early_warning)

        return full_reply


# Initialize chatbot
bot_brain = ChatBotBrain()