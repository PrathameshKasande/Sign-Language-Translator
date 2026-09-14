import os
import json
import datetime
import base64
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from predict_api import predict_frame

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sign_translator_secret_key_8892")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_FILE = os.path.join(BASE_DIR, 'users.json')

def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, 'w') as f:
            json.dump({}, f)
        return {}
    try:
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def log_login(email):
    log_path = os.path.join(BASE_DIR, "login_history.log")
    try:
        with open(log_path, "a") as log:
            log.write(f"{email} logged in at {datetime.datetime.now()}\n")
    except Exception:
        pass

@app.route('/', methods=['GET'])
def root():
    if session.get('logged_in'):
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('home'))
        
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        users = load_users()

        # Check if email exists and password matches
        if email in users and users[email] == password:
            log_login(email)
            session['logged_in'] = True
            session['email'] = email
            return redirect(url_for('home'))
        else:
            error = "Invalid credentials. Please register first if you do not have an account."

    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            error = "Email and password cannot be empty."
            return render_template('register.html', error=error)

        users = load_users()
        if email in users:
            error = "Email already registered. Please login."
            return render_template('register.html', error=error)

        # Dynamically store registered user credentials
        users[email] = password
        save_users(users)
        return redirect(url_for('login'))

    return render_template('register.html', error=error)

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html', user_email=session.get('email', 'pk@gmail.com'))

@app.route('/translate')
def translate():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('translate.html')

@app.route('/predict_frame', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        encoded = data.get("image", "")
        current_word = data.get("word", "")

        if "," in encoded:
            encoded = encoded.split(",")[1]

        if not encoded:
            return jsonify({"error": "No image payload"}), 400

        raw_bytes = base64.b64decode(encoded)
        result = predict_frame(raw_bytes, current_word=current_word)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)