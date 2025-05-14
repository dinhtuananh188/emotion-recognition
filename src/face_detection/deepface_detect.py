import os
import cv2
import shutil
import logging
import imghdr
from deepface import DeepFace
from retinaface import RetinaFace
from collections import Counter
from nudenet import NudeDetector

# Khởi tạo NudeDetector một lần
classifier = NudeDetector()

# Các nhãn NSFW cần loại bỏ
all_labels = [
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED", "BUTTOCKS_EXPOSED"
]

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Bản đồ cảm xúc sang chỉ số lớp
emotion_to_class = {
    'angry': 0, 'fear': 1, 'disgust': 2,
    'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
}

# Ngưỡng tin cậy mặc định
default_threshold = 0.90
lower_threshold = 0.80

# Đường dẫn
input_folder = "/mnt/d/Downloads/Compressed/Train/all_images"
output_folder = "/mnt/d/Downloads/Compressed/Train/processed"
output_images_folder = os.path.join(output_folder, "images")
output_labels_folder = os.path.join(output_folder, "labels")
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

def is_valid_image(file_path):
    try:
        if imghdr.what(file_path) is None:
            return False
        img = cv2.imread(file_path)
        if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
            return False
        return True
    except:
        return False

def get_current_emotion_counts():
    emotion_counts = Counter()
    processed_images = set()
    for filename in os.listdir(output_labels_folder):
        if not filename.endswith('.txt'):
            continue
        image_name = os.path.splitext(filename)[0]
        processed_images.add(image_name)
        try:
            with open(os.path.join(output_labels_folder, filename), 'r') as f:
                for line in f:
                    class_idx = int(line.strip().split()[0])
                    for emotion, idx in emotion_to_class.items():
                        if idx == class_idx:
                            emotion_counts[emotion] += 1
                            break
        except:
            continue
    return emotion_counts, processed_images

try:
    current_counts, processed_images = get_current_emotion_counts()
    logging.info("Current emotion counts:")
    for emotion, count in current_counts.items():
        logging.info(f"{emotion}: {count}")

    total_files = len([f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))])
    processed_files = 0
    saved_files = 0
    skipped_files = 0
    emotion_stats = Counter()

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue

        base_name = os.path.splitext(filename)[0]
        if base_name in processed_images:
            skipped_files += 1
            if skipped_files % 1000 == 0:
                logging.info(f"Skipped {skipped_files} previously processed images")
            continue

        file_path = os.path.join(input_folder, filename)
        try:
            if not is_valid_image(file_path):
                logging.warning(f"Invalid image: {filename}")
                os.remove(file_path)
                continue

            result_nude = classifier.detect(file_path)
            if result_nude and any(pred['class'] in all_labels for pred in result_nude):
                logging.info(f"NSFW image: {filename}")
                os.remove(file_path)
                continue

            img = cv2.imread(file_path)
            faces = RetinaFace.detect_faces(file_path)
            if not faces:
                logging.info(f"No face detected in: {filename}")
                os.remove(file_path)
                continue

            # Xác định ngưỡng tin cậy dựa vào số lượng khuôn mặt
            face_count = len(faces)
            # Nếu ảnh có từ 5 khuôn mặt trở lên, giảm ngưỡng tin cậy
            threshold_to_use = lower_threshold if face_count >= 5 else default_threshold
            logging.info(f"Image {filename} has {face_count} faces. Using threshold: {threshold_to_use}")

            valid_faces = []

            for idx, (face_key, face_data) in enumerate(faces.items()):
                x1, y1, x2, y2 = face_data['facial_area']
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                try:
                    result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                    face_emotion = result[0] if isinstance(result, list) else result
                    emotion_data = face_emotion['dominant_emotion']
                    confidence = face_emotion['emotion'][emotion_data]
                    
                    # Sử dụng ngưỡng được xác định dựa trên số lượng khuôn mặt
                    if confidence < threshold_to_use:
                        continue
                    if emotion_data.lower() not in emotion_to_class:
                        continue

                    valid_faces.append({
                        'emotion_data': emotion_data,
                        'confidence': confidence,
                        'region': {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
                    })
                except Exception as e:
                    logging.warning(f"Emotion detection failed in face {idx+1} of {filename}: {str(e)}")
                    continue

            if valid_faces and len(valid_faces) == len(faces):
                shutil.copy2(file_path, os.path.join(output_images_folder, filename))
                label_path = os.path.join(output_labels_folder, base_name + ".txt")

                for face_idx, face_info in enumerate(valid_faces):
                    emotion_data = face_info['emotion_data']
                    confidence = face_info['confidence']
                    face_region = face_info['region']
                    class_index = emotion_to_class[emotion_data.lower()]
                    x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                    img_h, img_w = img.shape[:2]
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    bbox_width = w / img_w
                    bbox_height = h / img_h

                    mode = "w" if face_idx == 0 else "a"
                    with open(label_path, mode) as label_file:
                        label_file.write(f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n")

                    current_counts[emotion_data.lower()] += 1
                    emotion_stats[emotion_data.lower()] += 1
                    logging.info(f"Processed face {face_idx + 1} in {filename}: {emotion_data}, confidence={confidence:.2f}")

                saved_files += 1

            processed_files += 1
            if processed_files % 100 == 0:
                logging.info(f"\nProcessed {processed_files}/{total_files} files, Saved {saved_files} files")
                for emotion, count in current_counts.items():
                    logging.info(f"{emotion}: {count}")

        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            continue

    logging.info("\nProcessing completed.")
    logging.info(f"- Total files processed: {processed_files}/{total_files}")
    logging.info(f"- Skipped previously processed files: {skipped_files}")
    logging.info(f"- New files saved: {saved_files}")
    logging.info("\nNew emotions this run:")
    for emotion, count in emotion_stats.items():
        logging.info(f"{emotion}: {count}")
    logging.info("\nFinal emotion counts:")
    for emotion, count in current_counts.items():
        logging.info(f"{emotion}: {count}")

except Exception as e:
    logging.error(f"Fatal error: {str(e)}")