
from extensions import db  # <--- Change this line
from flask_login import UserMixin
from datetime import datetime

# (Keep the rest of your User and ChatSession classes exactly the same)



class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False) # Will be hashed for privacy
    
    # The "Digital Twin" state (Aggregated from history)
    current_mood_score = db.Column(db.Float, default=0.0)
    last_assessment = db.Column(db.String(200), default="No data yet")
    
    # Relationship: One user has many chat sessions
    chats = db.relationship('ChatSession', backref='owner', lazy=True)

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Raw Data for Analysis
    full_conversation = db.Column(db.Text, nullable=False)
    
    # Analysis Results (The output you requested)
    sentiment_score = db.Column(db.Float)
    assessment_report = db.Column(db.Text)
    
    # Temporal Analysis (Time & Frequency)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    session_duration_minutes = db.Column(db.Integer)

    def __repr__(self):
        return f'<ChatSession {self.timestamp}>'
