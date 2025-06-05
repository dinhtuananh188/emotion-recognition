import uuid
from flask import Flask, jsonify, request, render_template, Response, session
from yolo_module import process_frame_with_yolo
import cv2
import numpy as np
from gemini_module import handle_request
import logging
import os
from stt_module import transcribe_audio, convert_to_wav_mono_16k
from waitress import serve
import base64
import shutil
import time

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '123456'

# Ensure the 'uploads' directory exists
uploads_dir = 'webapp/uploads'
audio_dir = 'webapp/static/audio'

@app.route('/api/clear_temp_dirs', methods=['DELETE'])
def clear_temp_dirs():
    try:
        # Xác định thư mục cần xóa
        for folder in ['upload_dir', 'audio_dir']:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                os.makedirs(folder)  # Tạo lại thư mục trống

        return {'status': 'success'}, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/')
def index():
    # Clear session data
    session.clear()
    session['chat_history'] = []
    session['sent_videos'] = []
    session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/yolo', methods=['POST'])
def process_yolo():
    try:
        # Get base64 image data from request
        image_data = request.json.get('frame')
        if not image_data:
            return jsonify({"error": "No frame data provided"}), 400

        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_frame, emotion_label = process_frame_with_yolo(frame)
        if processed_frame is None:
            return jsonify({"error": "Failed to process frame with YOLO"}), 500

        # Add processed frame to result
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_data = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "processed_frame": processed_frame_data,
            "label": emotion_label
        })
    except Exception as e:
        logging.error(f"Error in process_yolo: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/track_yolo', methods=['POST'])
def track_yolo():
    try:
        logging.debug("Starting YOLO tracking...")

        # Get base64 image data from request
        image_data = request.json.get('frame')
        if not image_data:
            return jsonify({"error": "No frame data provided"}), 400

        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Process frame with YOLO
        processed_frame, emotion_label = process_frame_with_yolo(frame)
        if processed_frame is None:
            return jsonify({"error": "Failed to process frame with YOLO"}), 500

        # Add processed frame to result
        _, buffer = cv2.imencode('.jpg', processed_frame)
        result = {
            "processed_frame": base64.b64encode(buffer).decode('utf-8'),
            "label": emotion_label
        }
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in /api/track_yolo: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/gemini', methods=['POST'])
def gemini_request():
    try:
        logging.debug("Received request for Gemini API.")
        user_input = request.json.get('user_input')
        if not user_input:
            logging.warning("Missing user input in Gemini request.")
            return jsonify({"error": "Missing user input."}), 400

        result = handle_request(user_input, session)
        tts_url = result.get("tts_url")
        if tts_url:
            tts_url = f"{tts_url}?t={int(time.time())}"  # Add cache-busting parameter

        resp = jsonify({
            "response": result.get("response"),
            "tts_url": tts_url
        })
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp
    except Exception as e:
        logging.error(f"Error in /api/gemini: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/audio', methods=['POST'])
def handle_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided."}), 400

        audio_file = request.files['audio']
        wav_path = os.path.join(uploads_dir, f"{uuid.uuid4()}.wav")
        audio_file.save(wav_path)

        file_size = os.path.getsize(wav_path)
        logging.debug(f"Saved WAV file at {wav_path} with size {file_size} bytes.")

        # Convert audio to mono, 16kHz
        converted_path = wav_path.replace(".wav", "_converted.wav")
        convert_to_wav_mono_16k(wav_path, converted_path)

        # Transcribe the converted audio
        token = os.getenv('VIETTEL_TTS_API_KEY')
        transcribed_text = transcribe_audio(converted_path, token)

        if not transcribed_text:
            return jsonify({"error": "Failed to transcribe audio."}), 500

        return jsonify({"message": "Audio transcribed successfully.", "text": transcribed_text})
    except Exception as e:
        logging.error(f"Error handling audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/session_debug')
def session_debug():
    return jsonify({
        "chat_history": session.get('chat_history', []),
        "sent_videos": session.get('sent_videos', []),
        "session_id": session.get('session_id')
    })

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=2402)