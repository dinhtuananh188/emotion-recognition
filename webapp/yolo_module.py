import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from ultralytics import YOLO
import time
from collections import Counter

# Load model và font
emotion_model = YOLO("../runs/detect/data_20k/weights/best.pt")
font = ImageFont.truetype("arial.ttf", 20)

# MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

emotion_mapping = {
    "Angry": "Giận dữ",
    "Contempt": "Khinh thường",
    "Disgust": "Chán ghét",
    "Happy": "Vui vẻ",
    "Neutral": "Bình thường",
    "Sad": "Buồn bã",
    "Surprise": "Ngạc nhiên",
}

emotion_colors = {
    "Angry": (255, 0, 0),
    "Contempt": (0, 255, 255),
    "Disgust": (128, 0, 128),
    "Happy": (0, 255, 0),
    "Neutral": (255, 192, 203),
    "Sad": (255, 0, 0),
    "Surprise": (218, 165, 32),
}


def process_frame_with_yolo(frame):
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    results = face_detector.process(rgb_frame)
    last_emotion = None

    if not results.detections:
        return frame, None

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * width)
        y1 = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        x2, y2 = x1 + w, y1 + h

        # Giới hạn trong khung hình
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Resize nhỏ để YOLO xử lý nhanh hơn (nếu cần)
        resized_crop = cv2.resize(face_crop, (224, 224))

        results = emotion_model(resized_crop, verbose=False)
        if not results or len(results[0].boxes.data) == 0:
            continue

        # Lấy kết quả emotion có confidence cao nhất
        top_box = max(results[0].boxes.data.tolist(), key=lambda b: b[4])
        _, _, _, _, score, class_id = top_box

        if score < 0.5:
            continue

        english_emotion = emotion_model.names[int(class_id)]
        viet_emotion = emotion_mapping.get(english_emotion, "Không xác định")
        box_color = emotion_colors.get(english_emotion, (128, 128, 128))

        # Lưu lại emotion cuối cùng được phát hiện
        last_emotion = viet_emotion

        # Vẽ bounding box quanh khuôn mặt gốc
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Vẽ nhãn tiếng Anh và tiếng Việt
        label_text = f"{english_emotion} {score:.1f}"
        viet_text = f"{viet_emotion}"

        viet_bbox = draw.textbbox((x1, y1 - 50), viet_text, font=font)
        draw.rectangle([viet_bbox[0]-2, viet_bbox[1]-2, viet_bbox[2]+2, viet_bbox[3]+2], fill=box_color)
        draw.text((x1, y1 - 50), viet_text, font=font, fill=(255, 255, 255))

        text_bbox = draw.textbbox((x1, y1 - 25), label_text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=box_color)
        draw.text((x1, y1 - 25), label_text, font=font, fill=(255, 255, 255))

    processed_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return processed_frame, last_emotion