import cv2
import mediapipe as mp
import os
from nudenet import NudeDetector
import imghdr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Face Detection with model selection
mp_face_detection = mp.solutions.face_detection
# Use model_selection=0 for short-range detection (within 2 meters)
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # Use short-range model
    min_detection_confidence=0.5
)

# Initialize NudeNet detector
detector = NudeDetector()

folder_path = "FEC_dataset/all_images"
label_folder = "FEC_dataset/labels"
os.makedirs(label_folder, exist_ok=True)

def is_valid_image(file_path):
    """Check if the file is a valid image and not corrupted."""
    try:
        # Check if file is actually an image
        if imghdr.what(file_path) is None:
            return False
        
        # Try to read the image
        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            return False
            
        # Check if image has valid dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False
            
        return True
    except Exception:
        return False

try:
    total_files = len([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))])
    processed_files = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            os.remove(file_path)
            
            continue
        
        try:
            # Validate image before processing
            if not is_valid_image(file_path):
                logging.warning(f"Invalid or corrupted image: {filename}")
                os.remove(file_path)
                continue

            # Check for inappropriate content
            result = detector.detect(file_path)
            if result and any(pred['class'] in ['EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F', 'EXPOSED_BUTTOCKS_F', 'EXPOSED_ANUS_F'] 
                            for pred in result):
                logging.info(f"Removing inappropriate image: {filename}")
                os.remove(file_path)
                continue

            img = cv2.imread(file_path)
            if img is None:
                logging.warning(f"Could not read image: {filename}")
                os.remove(file_path)
                continue
            
            # Convert the BGR image to RGB and ensure proper format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = face_detection.process(img_rgb)
            
            if not results.detections:
                logging.info(f"No face detected in: {filename}")
                os.remove(file_path)
                continue
            
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")
            with open(label_path, "w") as label_file:
                h, w, _ = img.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x_center = (bbox.xmin + bbox.width / 2) * w
                    y_center = (bbox.ymin + bbox.height / 2) * h
                    bbox_width = bbox.width * w
                    bbox_height = bbox.height * h
                    label_file.write(f"0 {x_center / w} {y_center / h} {bbox_width / w} {bbox_height / h}\n")
            
            processed_files += 1
            if processed_files % 100 == 0:
                logging.info(f"Processed {processed_files}/{total_files} files")
        
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            continue

finally:
    # Release resources
    face_detection.close()
    logging.info(f"Processing completed. Total files processed: {processed_files}/{total_files}")
