import os
import cv2
import numpy as np
from retinaface import RetinaFace
from tqdm import tqdm
import argparse
import imghdr
from pathlib import Path

def is_valid_image(file_path):
    """Check if file is a valid image."""
    try:
        # Check if file is an image using imghdr
        img_type = imghdr.what(file_path)
        if img_type is None:
            return False
        
        # Try to open the image with OpenCV
        img = cv2.imread(file_path)
        if img is None:
            return False
            
        # Check if image has valid dimensions
        height, width = img.shape[:2]
        if height <= 0 or width <= 0:
            return False
            
        return True
    except Exception:
        return False

def detect_faces(image_path):
    """Detect faces in an image using RetinaFace."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        faces = RetinaFace.detect_faces(img)
        return faces
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_directory(input_dir):
    """Process directory and remove invalid files and images without faces."""
    total_files = 0
    removed_files = 0
    invalid_images = 0
    no_faces = 0
    
    # Get all files in directory
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    total_files = len(all_files)
    
    print(f"Found {total_files} files to process")
    
    # Process each file
    for file_name in tqdm(all_files, desc="Processing files"):
        file_path = os.path.join(input_dir, file_name)
        
        # Check if it's a valid image
        if not is_valid_image(file_path):
            print(f"Removing invalid file: {file_name}")
            os.remove(file_path)
            removed_files += 1
            invalid_images += 1
            continue
        
        # Detect faces in valid images
        faces = detect_faces(file_path)
        
        # Remove images without faces
        if faces is None or len(faces) == 0:
            print(f"Removing image without faces: {file_name}")
            os.remove(file_path)
            removed_files += 1
            no_faces += 1
        else:
            # Optional: Print information about detected faces
            print(f"Found {len(faces)} faces in {file_name}")
            for face_idx, face_info in faces.items():
                score = face_info.get("score", 0)
                if score < 0.9:  # Remove low confidence detections
                    print(f"Removing image with low confidence face detection: {file_name}")
                    os.remove(file_path)
                    removed_files += 1
                    no_faces += 1
                    break
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Invalid/corrupted images removed: {invalid_images}")
    print(f"Images without faces removed: {no_faces}")
    print(f"Total files removed: {removed_files}")
    print(f"Remaining files: {total_files - removed_files}")

def main():
    parser = argparse.ArgumentParser(description='Clean directory by removing invalid files and images without faces')
    parser.add_argument('input_dir', help='Input directory containing images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    process_directory(args.input_dir)
    print("Processing completed!")

if __name__ == "__main__":
    main() 