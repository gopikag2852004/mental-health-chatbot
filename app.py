import os
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from extensions import db
from models import User, ChatSession
from ai_engine.analyzer import process_message
from ai_engine.chatbot import bot_brain

app = Flask(__name__)

# Setup Database Path
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'data', 'user_history.db')

app.config['SECRET_KEY'] = 'dev-key-123-digital-twin'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

with app.app_context():
    if not os.path.exists(os.path.join(basedir, 'data')):
        os.makedirs(os.path.join(basedir, 'data'))
    db.create_all()

@app.route('/')
@login_required
def home():
    return render_template('dashboard.html', current_user=current_user)

# --- ADDED REGISTRATION ROUTE ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check if user already exists
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already taken. Please choose another.', 'error')
            return redirect(url_for('register'))
            
        # Hash password and create user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created! You can now login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and bcrypt.check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid login credentials', 'error')
    return render_template('login.html')

@app.route('/get_response', methods=['POST'])
@login_required
def get_response():
    try:
        user_text = request.json.get('message')
        # Analyze with AI
        result = process_message(user_text, user_id=current_user.id)
        # Get Chatbot Response
        reply = bot_brain.get_reply(user_text)

        # Save to Database
        new_chat = ChatSession(
            user_id=current_user.id,
            full_conversation=user_text,
            sentiment_score=result["severity_score"],
            assessment_report=result["diagnosis"]
        )
        db.session.add(new_chat)
        db.session.commit()

        return jsonify({
            "reply": reply,
            "mood_score": result["severity_score"],
            "assessment": result["status"],
            "diagnosis": result["diagnosis"]
        })
    except Exception as e:
        print(f"❌ Backend Error: {e}")
        return jsonify({"reply": "Neural link error.", "error": str(e)}), 500

@app.route('/progress')
@login_required
def progress():
    chats = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.timestamp).all()
    scores = [c.sentiment_score for c in chats]
    times = [c.timestamp.strftime("%b %d %H:%M") for c in chats]
    return render_template('progress.html', scores=scores, times=times)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True, port=5000)