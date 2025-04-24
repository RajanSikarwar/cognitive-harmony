# app.py - Main Flask Application
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
import psycopg2
from psycopg2 import pool
from datetime import datetime, timedelta
import uuid
import json

# Import ONLY the chat function needed for direct chat
# Note: Ensure llm.py and its dependencies (models) are correctly loaded

# Removed detect_mood, personalize_questions, generate_final_message imports as they are not used in the simplified chat

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session expires after 7 days

# Import and register the Blueprint from llm.py
from llm import llm_api # Or whatever you name the blueprint variable in llm.py
app.register_blueprint(llm_api, url_prefix='/api') # Prefixes all routes in the blueprint with /api

# --- Database Setup (keeping previous improvements) ---
try:
    db_password = os.environ.get("DB_PASSWORD", "password") 
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20,
        user="postgres",
        password=db_password, 
        host="localhost",
        port="5432",
        database="cognitive_harmony"
    )
    print("Database connection pool created successfully")
except Exception as e:
    print(f"Error creating connection pool: {e}")
    connection_pool = None

def get_db_connection():
    if connection_pool:
        try:
            return connection_pool.getconn()
        except Exception as e:
             print(f"Error getting DB connection from pool: {e}")
             return None
    return None

def return_db_connection(conn):
    if connection_pool and conn:
        connection_pool.putconn(conn)

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Users table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        fullname VARCHAR(100) NOT NULL,
                        email VARCHAR(100) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    )
                ''')
                # Journal entries table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS journal_entries (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE, 
                        content TEXT NOT NULL,
                        mood VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                # Mood history table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS mood_history (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE, 
                        mood VARCHAR(50) NOT NULL,
                        intensity INTEGER CHECK (intensity BETWEEN 1 AND 10),
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                print("Database tables checked/initialized successfully")
        except Exception as e:
            if conn: conn.rollback()
            print(f"Error initializing database: {e}")
        finally:
            if conn: return_db_connection(conn)

with app.app_context():
     try:
         init_db()
     except Exception as e:
         print(f"Failed to initialize DB on startup: {e}")

# --- Routes ---

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('dashboard'))
    if request.method == 'GET': return render_template('register.html')

    data = request.get_json() if request.is_json else request.form
    fullname = data.get('fullname')
    email = data.get('email')
    password = data.get('password')
    
    if not fullname or not email or not password:
        message = "All fields are required"
        return jsonify({"success": False, "message": message}) if request.is_json else (flash(message, "error"), redirect(url_for('register')))
        
    password_hash = generate_password_hash(password)
    conn = get_db_connection()
    if not conn:
        message = "Database connection error"
        return jsonify({"success": False, "message": message}) if request.is_json else (flash(message, "error"), redirect(url_for('register')))
        
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                message = "Email already registered"
                return jsonify({"success": False, "message": message}) if request.is_json else (flash(message, "error"), redirect(url_for('register')))
                
            cur.execute(
                "INSERT INTO users (fullname, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
                (fullname, email, password_hash)
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            
            session['user_id'] = user_id
            session['user_name'] = fullname
            session['user_email'] = email
            session.permanent = True 
            
            message = "Registration successful"
            return jsonify({"success": True, "message": message}) if request.is_json else (flash(message, "success"), redirect(url_for('dashboard')))
    except Exception as e:
        if conn: conn.rollback()
        print(f"Error registering user: {e}")
        message = "Registration failed due to server error"
        return jsonify({"success": False, "message": message}) if request.is_json else (flash(message, "error"), redirect(url_for('register')))
    finally:
        if conn: return_db_connection(conn)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('dashboard'))
    if request.method == 'GET': return render_template('login.html')

    data = request.get_json() if request.is_json else request.form
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        message = "Email and password are required"
        return jsonify({"success": False, "message": message}) if request.is_json else (flash(message, "error"), redirect(url_for('login')))
        
    conn = get_db_connection()
    if not conn:
        message = "Database connection error"
        return jsonify({"success": False, "message": message}) if request.is_json else (flash(message, "error"), redirect(url_for('login')))
        
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, fullname, email, password_hash FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            
            if user and check_password_hash(user[3], password):
                cur.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s", (user[0],))
                conn.commit()
                
                session['user_id'] = user[0]
                session['user_name'] = user[1]
                session['user_email'] = user[2]
                session.permanent = True 
                
                message = "Login successful"
                return jsonify({"success": True, "message": message}) if request.is_json else redirect(url_for('dashboard'))
            else:
                message = "Invalid email or password"
                return jsonify({"success": False, "message": message}) if request.is_json else (flash(message, "error"), redirect(url_for('login')))
    except Exception as e:
        print(f"Error during login: {e}")
        message = "Login failed due to server error"
        return jsonify({"success": False, "message": message}) if request.is_json else (flash(message, "error"), redirect(url_for('login')))
    finally:
        if conn: return_db_connection(conn)


@app.route('/logout')
def logout():
    # Clear only user-specific session data
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('user_email', None)
    # Removed assessment state clearing as it's no longer used
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for('login'))
    user_name = session.get('user_name', 'User') 
    return render_template('dashboard.html', user_name=user_name)

# --- SIMPLIFIED CHAT ROUTE ---
# @app.route('/chat', methods=['POST'])
# def chat():
#     if 'user_id' not in session:
#         return jsonify({"error": "Please log in to continue the chat"}), 401
        
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No data received"}), 400
            
#         user_message = data.get('message')
#         if not user_message or not isinstance(user_message, str):
#             return jsonify({"error": "Invalid or missing message"}), 400
            
#         print(f"Processing chat message: '{user_message[:50]}...'")
        
#         # Call the LLM for response
#         ai_response = generate_chat_response(user_message)
        
#         if not ai_response:
#             return jsonify({"error": "No response generated"}), 500
            
#         print(f"AI response generated: '{ai_response[:50]}...'")
#         return jsonify({"reply": ai_response})
        
#     except Exception as e:
#         print(f"Error in chat route: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500
# --- END SIMPLIFIED CHAT ROUTE ---

@app.route('/resources')
def resources():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('resources.html', user_name=session.get('user_name'))

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# Remove the setup_templates route if not needed for deployment
# @app.route('/setup_templates')
# def setup_templates():
#      return "Template setup route (for development)"

# Run the application
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Debug should generally be False in production
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
