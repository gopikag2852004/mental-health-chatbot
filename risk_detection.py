class RiskDetector:

    def __init__(self):
        self.mood_history = []

    def add_mood(self, mood):

        self.mood_history.append(mood)

        if len(self.mood_history) > 15:
            self.mood_history.pop(0)

    def detect_risk(self):

        negative = ["sad", "anxiety", "stress"]

        count = sum(1 for m in self.mood_history if m in negative)

        if count >= 6:
            return True

        return False