import cv2
import os
from ultralytics import YOLO
import shutil
from tqdm import tqdm
import numpy as np

# Load YOLO emotion model
emotion_model = YOLO(r"D:\Downloads\Compressed\Train\runs\detect\train15\weights\last.pt")

def create_directories(base_dir):
    """Create necessary directories for output"""
    dirs = {
        'need_detect': os.path.join(base_dir, 'Need_to_detect'),
        'need_detect_labels': os.path.join(base_dir, 'Need_to_detect', 'labels'),
        'need_detect_images': os.path.join(base_dir, 'Need_to_detect', 'images'),
        'done': os.path.join(base_dir, 'Done'),
        'done_labels': os.path.join(base_dir, 'Done', 'labels'),
        'done_images': os.path.join(base_dir, 'Done', 'images')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def read_label_file(label_path):
    """Read YOLO format label file"""
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    labels = []
    for line in lines:
        values = line.strip().split()
        if len(values) == 5:
            labels.append({
                'class': int(values[0]),
                'x_center': float(values[1]),
                'y_center': float(values[2]),
                'width': float(values[3]),
                'height': float(values[4])
            })
    return labels

def crop_face(image, bbox):
    """Crop face from image using normalized YOLO coordinates"""
    ih, iw = image.shape[:2]
    x_center, y_center = bbox['x_center'] * iw, bbox['y_center'] * ih
    width, height = bbox['width'] * iw, bbox['height'] * ih
    
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    
    return image[y1:y2, x1:x2]

def process_images_with_labels(input_images_dir, input_labels_dir, output_base_dir):
    """Process images with existing labels and categorize results"""
    dirs = create_directories(output_base_dir)
    
    # Get list of images with corresponding labels
    image_files = [f for f in os.listdir(input_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in tqdm(image_files, desc="Processing images"):
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(input_images_dir, image_file)
        label_path = os.path.join(input_labels_dir, f"{base_name}.txt")
        
        # Skip if no label file exists
        if not os.path.exists(label_path):
            continue
        
        # Read image and labels
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_file}")
            continue
            
        labels = read_label_file(label_path)
        if not labels:
            continue
        
        needs_manual_detection = False
        new_labels = []
        
        for label in labels:
            # Crop face using label coordinates
            face_crop = crop_face(image, label)
            
            # Detect emotion on cropped face
            results = emotion_model.predict(source=face_crop, save=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Use the first detection's class (assuming highest confidence)
                new_class = int(results[0].boxes[0].cls[0])
                if new_class == 9:  # If still class 9, mark for manual detection
                    needs_manual_detection = True
                label['class'] = new_class
            else:
                needs_manual_detection = True
            
            new_labels.append(label)
        
        # Determine destination directories based on detection results
        if needs_manual_detection:
            dest_img_dir = dirs['need_detect_images']
            dest_label_dir = dirs['need_detect_labels']
        else:
            dest_img_dir = dirs['done_images']
            dest_label_dir = dirs['done_labels']
        
        # Save image and updated labels
        shutil.copy2(image_path, os.path.join(dest_img_dir, image_file))
        
        # Write new labels
        with open(os.path.join(dest_label_dir, f"{base_name}.txt"), 'w') as f:
            for label in new_labels:
                f.write(f"{label['class']} {label['x_center']} {label['y_center']} {label['width']} {label['height']}\n")

def main():
    print("=== Starting batch emotion detection process ===")
    try:
        # Update these paths according to your directory structure
        input_images_dir = "FEC_dataset/images"
        input_labels_dir = "FEC_dataset/labels"
        output_base_dir = "FEC_dataset/processed"
        
        process_images_with_labels(input_images_dir, input_labels_dir, output_base_dir)
        print("\nProcessing completed!")
        print(f"Check 'Need_to_detect' directory for images requiring manual detection")
        print(f"Check 'Done' directory for successfully processed images")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
