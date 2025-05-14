import os
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define emotion mapping (same as in deepface_detect.py)
class_to_emotion = {
    0: 'angry',
    1: 'contempt',
    2: 'disgust',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

def count_emotions(labels_dir):
    """Count the number of faces for each emotion in the dataset."""
    # Initialize counters
    emotion_counts = Counter()  # Count by emotion
    total_faces = 0
    total_images = 0
    images_with_multiple_faces = 0
    
    try:
        # Iterate through all label files
        for filename in os.listdir(labels_dir):
            if not filename.endswith('.txt'):
                continue
                
            file_path = os.path.join(labels_dir, filename)
            faces_in_image = 0
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                # Count faces in this image
                faces_in_image = len(lines)
                total_faces += faces_in_image
                
                if faces_in_image > 0:
                    total_images += 1
                
                if faces_in_image > 1:
                    images_with_multiple_faces += 1
                
                # Count emotions
                for line in lines:
                    try:
                        # YOLO format: class x_center y_center width height
                        class_idx = int(line.strip().split()[0])
                        emotion = class_to_emotion.get(class_idx, 'unknown')
                        emotion_counts[emotion] += 1
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error parsing line in {filename}: {str(e)}")
                        
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
                continue
        
        # Print statistics
        logging.info("\n=== Dataset Statistics ===")
        logging.info(f"Total images processed: {total_images}")
        logging.info(f"Total faces detected: {total_faces}")
        logging.info(f"Images with multiple faces: {images_with_multiple_faces}")
        logging.info("\n=== Emotion Distribution ===")
        
        # Calculate percentages and print emotion counts
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / total_faces * 100) if total_faces > 0 else 0
            logging.info(f"{emotion.capitalize()}: {count} faces ({percentage:.2f}%)")
            
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    # Use the same directory structure as deepface_detect.py
    labels_dir = "FEC_dataset/fear_processed/labels"
    
    if not os.path.exists(labels_dir):
        logging.error(f"Labels directory not found: {labels_dir}")
    else:
        count_emotions(labels_dir) 