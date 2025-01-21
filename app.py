from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import subprocess

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Database URL
db = SQLAlchemy(app)


# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)


# Initialize the database
with app.app_context():
    db.create_all()


@app.route('/')
def start():
    return redirect(url_for('register'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"status": "error", "message": "User already exists"}), 400

    try:
        new_user = User(email=email, password=password)  # Hash passwords in production!
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"status": "success", "message": "User registered successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    # Placeholder for POST logic if needed
    return redirect(url_for('home'))


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/start-posture', methods=['POST'])
def start_posture():
    try:
        # Start the posture monitoring script
        subprocess.Popen(["python", "test.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify({"status": "success", "message": "Posture monitoring started."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
