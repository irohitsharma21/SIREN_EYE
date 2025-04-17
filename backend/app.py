import os
import cv2
import torch
import numpy as np
import librosa
from flask import Flask, request, jsonify
from keras.models import load_model
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import moviepy.editor as mp

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("models/best.pt")  # Ensure this path is correct in Render
audio_model = load_model("models/ambulance_siren_model.h5")

# Constants
SAMPLE_RATE = 22050

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is running!"})

@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(frame, conf=0.65, device=device)
        detected = False

        for result in results:
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = result.boxes.conf[i].item() * 100

                if confidence > 65:
                    detected = True
                    detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": confidence})

        if detected:
            if verify_siren(filepath):
                cap.release()
                return jsonify({"status": "Emergency Ambulance Detected", "detections": detections})

    cap.release()
    return jsonify({"status": "No active emergency ambulance detected", "detections": detections})


def verify_siren(video_path):
    """Extracts audio and checks for siren sound."""
    temp_audio_path = extract_audio(video_path)
    if temp_audio_path:
        return classify_audio(temp_audio_path)
    return False


def extract_audio(video_path):
    """Extracts a short audio segment from the video."""
    try:
        video = mp.VideoFileClip(video_path)
        audio_path = os.path.join(UPLOAD_FOLDER, "temp_audio.wav")
        video.audio.write_audiofile(audio_path, fps=SAMPLE_RATE)
        return audio_path
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None


def classify_audio(audio_path):
    """Classifies whether the extracted audio contains an ambulance siren."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 128 - log_mel_spec.shape[1])), mode='constant')
        mel_spec_reshaped = np.expand_dims(log_mel_spec, axis=-1)
        prediction = audio_model.predict(np.expand_dims(mel_spec_reshaped, axis=0))
        return prediction[0][0] > 0.5
    except Exception as e:
        print(f"Audio processing error: {e}")
        return False


# ❗️DO NOT include `app.run()` for Render (Gunicorn handles it)
# Leave this part out so gunicorn can load `app` correctly
