import os
import glob
from PIL import Image
from pathlib import Path
import json
import shutil
from ultralytics import YOLO  # YOLOv11 library
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import concurrent.futures
import time
import numpy as np

# Mapping of emotions according to data.yaml
EMOTION_MAPPING = {
    "Anger": 0,
    "Contempt": 1,
    "Disgust": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6
}

class QwenAPI:
    def __init__(self, model_path="Qwen2-VL-2B-Instruct"):
        # Load the Qwen model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def analyze_emotion(self, image_path):
        """
        Analyze the emotion in the given image using Qwen2-VL-2B-Instruct.
        Args:
            image_path: Path to the image file.
        Returns:
            Emotion label as a string or None if an error occurs.
        """
        def process_image():
            # Path to the reference sample
            
            # Qwen processing
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "If the person in the image raises one side of their mouth, return 'Contempt'. "
                                "If they do not raise either side, return 'None'. Return only the classification, nothing else."
                            ),
                        },
                        {"type": "image", "image": image_path},  # Target image
                    ],
                }
            ]


            # Prepare inputs for the model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Generate output
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0].strip()

        try:
            result = process_image()
            #time.sleep(2)
            return result
        except Exception as e:
            print(f"Error while processing {image_path}: {str(e)}")
            return None

def crop_face(image, x_center, y_center, width, height):
    """
    Crop face from image using YOLO format coordinates
    Args:
        image: PIL Image
        x_center, y_center: Center coordinates (normalized 0-1)
        width, height: Width and height of box (normalized 0-1)
    Returns:
        Cropped PIL Image
    """
    img_w, img_h = image.size
    
    # Convert normalized coordinates to pixel coordinates
    x_center = x_center * img_w
    y_center = y_center * img_h
    width = width * img_w
    height = height * img_h
    
    # Calculate box coordinates
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    
    # Add padding
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_w, x2 + padding)
    y2 = min(img_h, y2 + padding)
    
    # Crop and return
    return image.crop((x1, y1, x2, y2))

def fix_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that the image is not corrupted
            img = Image.open(image_path)  # Reopen the image after verification
            img.save(image_path)  # Save the image to remove invalid metadata
    except Exception as e:
        print(f"Error fixing image {image_path}: {str(e)}")
        return False  # Indicate that the image is invalid
    return True  # Indicate that the image is valid

def get_emotion_from_qwen(qwen_api, image_path, face_coords=None):
    try:
        with Image.open(image_path) as img:

            # Nếu có tọa độ khuôn mặt, cắt ảnh
            if face_coords:
                x, y, w, h = face_coords
                img = crop_face(img, x, y, w, h)
                cropped_image_path = "temp_cropped_image.jpg"
                img.save(cropped_image_path)
                image_path = cropped_image_path

            # Sử dụng concurrent.futures để đặt timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(qwen_api.analyze_emotion, image_path)
                try:
                    emotion = future.result(timeout=10)  # Timeout sau 10 giây
                    print(f"Qwen result for {image_path}: {emotion}")  # Log kết quả của Qwen
                except concurrent.futures.TimeoutError:
                    print(f"Timeout: Qwen did not respond for {image_path} within 10 seconds")
                    # Xóa ảnh tạm thời nếu có
                    if face_coords and os.path.exists(cropped_image_path):
                        os.remove(cropped_image_path)
                    return None  # Trả về None để tiếp tục xử lý ảnh khác

        # Mapping emotion từ kết quả Qwen
        for key in EMOTION_MAPPING.keys():
            if key.lower() in emotion.lower():
                return EMOTION_MAPPING[key]

        print(f"Warning: Could not map Qwen response '{emotion}' to known emotions")
        return None

    except Exception as e:
        print(f"Error processing face: {str(e)}")
        return None

def create_emotion_folders(base_dir):
    categorized_dir = os.path.join(base_dir, 'categorized_emotions')
    os.makedirs(categorized_dir, exist_ok=True)
    for emotion in EMOTION_MAPPING.keys():
        emotion_dir = os.path.join(categorized_dir, emotion)
        os.makedirs(os.path.join(emotion_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(emotion_dir, 'labels'), exist_ok=True)
    return categorized_dir

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1, box2: Bounding boxes in the format (x1, y1, x2, y2)
    Returns:
        IoU value (float)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def process_directory(qwen_api, base_dir, yolo_model, categorized_dir):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    progress_file = os.path.join(base_dir, 'processing_progress.json')
    
    # Load progress from previous run
    processed_files = {}
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_files = json.load(f)
    
    # Get all image files
    image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + \
                 glob.glob(os.path.join(images_dir, '*.png'))
    
    total_files = len(image_files)
    for idx, image_path in enumerate(image_files, 1):
        if not fix_image(image_path):  # Fix image metadata and check validity
            print(f"Skipping invalid image {image_path}")
            continue

        image_name = os.path.basename(image_path)

        # Skip if already processed
        if image_name in processed_files:
            print(f"Skipping already processed {image_name}")
            continue

        print(f"Processing {image_name}... ({idx}/{total_files})")

        # Run YOLO detection
        results = yolo_model(image_path)

        # Check if there are any detections
        if not results or not results[0].boxes:
            print(f"No detections found for {image_path}")
            continue

        detections = results[0].boxes.data.cpu().numpy()  # Get bounding boxes

        # Group overlapping bounding boxes
        grouped_faces = []
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, confidence, yolo_emotion = detection
            if confidence < 0.5:  # Skip low-confidence detections
                continue

            # Check if this bounding box overlaps with any existing group
            matched = False
            for group in grouped_faces:
                if iou((x1, y1, x2, y2), group['bbox']) > 0.5:  # IoU threshold
                    group['emotions'].add(int(yolo_emotion))
                    matched = True
                    break

            if not matched:
                grouped_faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'emotions': {int(yolo_emotion)}
                })

        all_faces_valid = True  # Flag to check if all faces are valid
        temp_valid_faces = 0    # Count of temporarily valid faces
        total_faces = len(grouped_faces)  # Total number of grouped faces

        for group in grouped_faces:
            x1, y1, x2, y2 = group['bbox']
            emotions = group['emotions']

            # Convert YOLO bbox to normalized format
            img = Image.open(image_path)
            img_w, img_h = img.size
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            # Get emotion from Qwen
            qwen_emotion = get_emotion_from_qwen(
                qwen_api, image_path, face_coords=(x_center, y_center, width, height)
            )

            if qwen_emotion is None:
                print(f"Skipping {image_path} due to timeout or error")
                continue  # Skip this face and move to the next one

            emotion_name = list(EMOTION_MAPPING.keys())[qwen_emotion]

            # Check if Qwen emotion matches any YOLO emotion
            if qwen_emotion is not None and qwen_emotion in emotions:
                temp_valid_faces += 1  # Temporarily valid face
            else:
                all_faces_valid = False  # Mark as invalid if any face is mismatched

        # Check if all faces are temporarily valid
        if temp_valid_faces == total_faces:
            for group in grouped_faces:
                x1, y1, x2, y2 = group['bbox']
                emotions = group['emotions']

                # Convert YOLO bbox to normalized format
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                qwen_emotion = get_emotion_from_qwen(
                    qwen_api, image_path, face_coords=(x_center, y_center, width, height)
                )
                emotion_name = list(EMOTION_MAPPING.keys())[qwen_emotion]
                emotion_folder = os.path.join(categorized_dir, emotion_name)

                # Move image and label to categorized folder
                shutil.copy(image_path, os.path.join(emotion_folder, 'images', image_name))
                label_path = os.path.join(labels_dir, Path(image_path).stem + '.txt')
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(emotion_folder, 'labels', Path(image_path).stem + '.txt'))

                # Create YOLO label file
                yolo_label_path = os.path.join(emotion_folder, 'labels', Path(image_path).stem + '.txt')
                with open(yolo_label_path, 'w') as f:
                    f.write(f"{qwen_emotion} {x_center} {y_center} {width} {height}\n")

                print(f"  Moved {image_name} to {emotion_name} folder")
        else:
            print(f"  Skipping {image_name} due to emotion mismatch or low confidence")
        
        # Mark as processed
        processed_files[image_name] = True
        with open(progress_file, 'w') as f:
            json.dump(processed_files, f)

def main():
    base_dir = "FEC_dataset"
    qwen_api = QwenAPI()

    # Load YOLO model
    yolo_model_path = r"D:\Downloads\Compressed\Train\runs\detect\train15\weights\best.pt"
    yolo_model = YOLO(yolo_model_path)

    # Create emotion folders
    categorized_dir = create_emotion_folders(base_dir)

    process_directory(qwen_api, base_dir, yolo_model, categorized_dir)
    print("Processing completed!")

if __name__ == "__main__":
    main()