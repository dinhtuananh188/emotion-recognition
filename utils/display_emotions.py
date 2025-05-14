import cv2
import os
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt

# Define paths
processed_folder = "FEC_dataset/deepface_processed"
images_folder = os.path.join(processed_folder, "images")
labels_folder = os.path.join(processed_folder, "labels")

# Define emotion mapping (same as in deepface_detect.py)
class_to_emotion = {
    0: 'angry',
    1: 'fear',
    2: 'disgust',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

def find_emotion_images(target_emotion, num_images=5):
    found_images = []
    
    print(f"\nSearching for {target_emotion} images...")
    target_class = None
    
    # Find the class index for target emotion
    for class_idx, emotion in class_to_emotion.items():
        if emotion == target_emotion:
            target_class = class_idx
            break
    
    # Go through all label files
    for label_file in os.listdir(labels_folder):
        if not label_file.endswith('.txt'):
            continue
            
        if len(found_images) >= num_images:
            break
            
        label_path = os.path.join(labels_folder, label_file)
        image_file = os.path.splitext(label_file)[0] + '.jpg'  # Assuming jpg extension
        image_path = os.path.join(images_folder, image_file)
        
        if not os.path.exists(image_path):
            # Try other extensions
            for ext in ['.jpeg', '.png', '.bmp', '.tiff']:
                image_path = os.path.join(images_folder, os.path.splitext(label_file)[0] + ext)
                if os.path.exists(image_path):
                    break
            else:
                continue
        
        try:
            # Read label file
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            # Check each face in the label file
            for line in lines:
                parts = line.strip().split()
                class_idx = int(parts[0])
                
                if class_idx == target_class:
                    # Get YOLO format coordinates
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert to pixel coordinates
                    img = cv2.imread(image_path)
                    img_height, img_width = img.shape[:2]
                    
                    x = int((x_center - width/2) * img_width)
                    y = int((y_center - height/2) * img_height)
                    w = int(width * img_width)
                    h = int(height * img_height)
                    
                    found_images.append({
                        'path': image_path,
                        'region': {'x': x, 'y': y, 'w': w, 'h': h}
                    })
                    print(f"Found {target_emotion} image: {image_file}")
                    break
                    
        except Exception as e:
            print(f"Error processing {label_file}: {str(e)}")
            continue
            
    return found_images

def display_emotion_examples():
    plt.figure(figsize=(20, 15))
    
    for i, emotion in enumerate(class_to_emotion.values()):
        # Find 5 images for current emotion
        emotion_images = find_emotion_images(emotion)
        
        for j, img_data in enumerate(emotion_images[:5]):
            try:
                # Read image
                img = cv2.imread(img_data['path'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract face region
                region = img_data['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2*padding)
                h = min(img.shape[0] - y, h + 2*padding)
                
                face = img[y:y+h, x:x+w]
                
                # Plot
                plt.subplot(7, 5, i*5 + j + 1)
                plt.imshow(face)
                plt.title(f"{emotion}")
                plt.axis('off')
                
            except Exception as e:
                print(f"Error displaying image: {str(e)}")
                continue
    
    plt.tight_layout()
    plt.savefig('processed_emotion_examples.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Starting emotion example search...")
    display_emotion_examples()
    print("\nCompleted! Check processed_emotion_examples.png for results") 