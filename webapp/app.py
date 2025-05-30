import uuid
import re
from flask import Flask, jsonify, request, render_template, Response, session
from camera_module import initialize_camera, capture_frame
from yolo_module import process_frame_with_yolo, track_labels_for_duration
import cv2
import numpy as np
from gemini_module import handle_request
import logging
import os
from stt_module import transcribe_audio
from pydub import AudioSegment
import time
from waitress import serve


def convert_to_wav_mono_16k(src_path, dst_path):
    """
    Converts any audio file to WAV format with mono channel and 16kHz sample rate.
    """
    sound = AudioSegment.from_file(src_path)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    sound.export(dst_path, format="wav")


# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '123456'

# Initialize camera
def safe_initialize_camera(retries=10, delay=1):
    for i in range(retries):
        try:
            cam = initialize_camera()
            if cam is not None:
                return cam
        except Exception as e:
            logging.warning(f"Retry {i+1}/{retries} - Camera init failed: {e}")
            time.sleep(delay)
    return None

# Gọi khi khởi tạo
camera = safe_initialize_camera()


# Ensure the 'uploads' directory exists
uploads_dir = 'webapp/uploads'
os.makedirs(uploads_dir, exist_ok=True)

# Clear all files in the 'uploads' directory
for file_name in os.listdir(uploads_dir):
    file_path = os.path.join(uploads_dir, file_name)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        logging.error(f"Error deleting file {file_path}: {e}")

@app.route('/')
def index():
    # Clear session data
    session.clear()
    session['chat_history'] = []
    session['sent_videos'] = []
    session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/camera', methods=['GET'])
def get_camera_frame():
    try:
        frame = capture_frame(camera)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        return jsonify({"frame": frame_data.hex()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/yolo', methods=['POST'])
def process_yolo():
    try:
        frame_hex = request.json.get('frame')
        frame_bytes = bytes.fromhex(frame_hex)
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

        processed_frame = process_frame_with_yolo(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_data = buffer.tobytes()
        return jsonify({"processed_frame": processed_frame_data.hex()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/gemini', methods=['POST'])
def gemini_request():
    try:
        logging.debug("Received request for Gemini API.")
        user_input = request.json.get('user_input')
        if not user_input:
            logging.warning("Missing user input in Gemini request.")
            return jsonify({"error": "Missing user input."}), 400

        logging.debug(f"User input: {user_input}")
        result = handle_request(user_input, session)
        
        return jsonify({
            "response": result.get("response"),
            "tts_url": result.get("tts_url")
        })
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

def generate_frames():
    while True:
        try:
            frame = capture_frame(camera)
            processed_frame = process_frame_with_yolo(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error: {e}")
            break

def frame_generator():
    while True:
        yield capture_frame(camera)

@app.route('/api/track_yolo', methods=['GET'])
def track_yolo():
    try:
        logging.debug("Starting YOLO tracking...")
        most_common_label = track_labels_for_duration(frame_generator(), duration=5)
        if not most_common_label:
            logging.warning("No labels detected during YOLO tracking.")
            return jsonify({"error": "Không phát hiện nhãn nào."}), 400

        logging.debug(f"Most common label detected: {most_common_label}")
        result = handle_request('Tôi đang cảm thấy ' + most_common_label, session)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in /api/track_yolo: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/session_debug')
def session_debug():
    return jsonify({
        "chat_history": session.get('chat_history', []),
        "sent_videos": session.get('sent_videos', []),
        "session_id": session.get('session_id')
    })

if __name__ == '__main__':
    serve(app, host="127.0.0.1", port=2402)