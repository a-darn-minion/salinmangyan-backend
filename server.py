from gtts import gTTS
from flask import Flask, request, send_file, jsonify
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wavfile
import numpy as np 
import os
import psycopg
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO

load_dotenv()


app = Flask(__name__)

# --- AI MODEL INITIALIZATION FOR HANUNÓ'O ---
model_name = "facebook/mms-tts-hnn"
model = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

USER_DB = "users.json"

# --- POSTGRESQL DATABASE CONFIGURATION ---

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from your .env file")


def get_db_connection():
    return psycopg.connect(
        DATABASE_URL,
        connect_timeout=10
    )


def create_users_table():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    profession VARCHAR(255),
                    address TEXT,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)

    print("PostgreSQL users table is ready.")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "SALINMANGYAN API"
    }), 200


@app.route("/register", methods=["POST"])
def register_user():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data received"}), 400

        email = data.get("email", "").strip().lower()
        name = data.get("name", "").strip()
        profession = data.get("profession", "").strip()
        address = data.get("address", "").strip()
        password = data.get("password", "")

        if not email:
            return jsonify({"error": "Email is required"}), 400

        if not password or len(password) < 8:
            return jsonify({
                "error": "Password must be at least 8 characters"
            }), 400

        password_hash = generate_password_hash(password)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users
                    (email, name, profession, address, password_hash)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    email,
                    name,
                    profession,
                    address,
                    password_hash
                ))

        print(f"Successfully registered user: {email}")

        return jsonify({
            "message": "Registration successful"
        }), 200

    except psycopg.errors.UniqueViolation:
        return jsonify({"error": "User already exists!"}), 400

    except Exception as e:
        print(f"CRITICAL REGISTRATION ERROR: {str(e)}")
        return jsonify({"error": "Registration failed"}), 500

@app.route("/login", methods=["POST"])
def login_user():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data received"}), 400

        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT name, password_hash
                    FROM users
                    WHERE email = %s
                """, (email,))

                user = cur.fetchone()

        if user and check_password_hash(user[1], password):
            print(f"User login authorized: {email}")

            return jsonify({
                "message": "Access granted",
                "name": user[0]
            }), 200

        return jsonify({
            "error": "Invalid email or password"
        }), 401

    except Exception as e:
        print(f"LOGIN ERROR: {str(e)}")
        return jsonify({"error": "Login failed"}), 500

# --- CLEANED AUDIO GENERATION ENDPOINT ---
@app.route("/translate-tts", methods=["POST"])
def translate_tts():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data received"}), 400

        text_to_speak = data.get("text", "").strip()
        mode = data.get("mode", "tag_to_hnn")

        print(f"Audio Request | Mode: {mode} | Text: '{text_to_speak}'")

        if not text_to_speak or text_to_speak.lower() in ["word not found", "not found"]:
            return jsonify({"error": "Valid text is required"}), 400

        audio_buffer = BytesIO()

        if "to_hnn" in mode:
            inputs = tokenizer(text_to_speak, return_tensors="pt")

            with torch.no_grad():
                output = model(**inputs).waveform[0]

            audio_data = output.detach().cpu().numpy()
            audio_data = (audio_data * 32767).astype(np.int16)

            wavfile.write(
                audio_buffer,
                rate=model.config.sampling_rate,
                data=audio_data
            )

            mimetype = "audio/wav"
            filename = "audio.wav"

        elif "to_eng" in mode:
            tts = gTTS(text=text_to_speak, lang="en")
            tts.write_to_fp(audio_buffer)

            mimetype = "audio/mpeg"
            filename = "audio.mp3"

        else:
            tts = gTTS(text=text_to_speak, lang="tl")
            tts.write_to_fp(audio_buffer)

            mimetype = "audio/mpeg"
            filename = "audio.mp3"

        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"TTS SYNTHESIS ERROR: {str(e)}")
        return jsonify({"error": "Audio generation failed"}), 500

# Initialize database when the application starts
create_users_table()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

