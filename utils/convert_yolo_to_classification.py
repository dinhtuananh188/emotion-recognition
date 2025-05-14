import os
import shutil
import yaml
from pathlib import Path
import glob

def create_classification_structure(source_dir, output_dir):
    """
    Convert YOLO format data to classification format
    
    Args:
        source_dir: Directory containing YOLO format data (train, valid, test)
        output_dir: Directory where classification data will be saved
    """
    # Read the data.yaml file to get class names
    yaml_path = os.path.join(source_dir, 'data.yaml')
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    
    # Create output directory structure
    for split in ['train', 'valid', 'test']:
        # Create main split directory
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class directories
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        print(f"Processing {split} split...")
        
        # Source directories
        images_dir = os.path.join(source_dir, split, 'images')
        labels_dir = os.path.join(source_dir, split, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Warning: {split} directory structure not found. Skipping...")
            continue
        
        # Get all image files
        image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + \
                      glob.glob(os.path.join(images_dir, '*.jpeg')) + \
                      glob.glob(os.path.join(images_dir, '*.png'))
        
        for img_path in image_files:
            # Get corresponding label file
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for {img_name}. Skipping...")
                continue
            
            # Read the label file to get the class
            with open(label_path, 'r') as f:
                # YOLO format: class_id x_center y_center width height
                # We only need the class_id (first number)
                line = f.readline().strip()
                if not line:
                    print(f"Warning: Empty label file for {img_name}. Skipping...")
                    continue
                
                class_id = int(line.split()[0])
                if class_id < 0 or class_id >= len(class_names):
                    print(f"Warning: Invalid class ID {class_id} for {img_name}. Skipping...")
                    continue
                
                class_name = class_names[class_id]
                
                # Copy image to the appropriate class directory
                dest_dir = os.path.join(output_dir, split, class_name)
                shutil.copy2(img_path, dest_dir)
        
        print(f"Completed processing {split} split.")

def move_valid_to_test(source_dir, output_dir):
    """
    Move all data from the valid directory to the test directory.
    
    Args:
        source_dir: Directory containing YOLO format data (train, valid, test)
        output_dir: Directory where classification data is saved
    """
    # Directories
    valid_images_dir = os.path.join(output_dir, 'valid')
    test_images_dir = os.path.join(output_dir, 'test')

    # Move images and labels from valid to test
    for class_name in os.listdir(valid_images_dir):
        valid_class_dir = os.path.join(valid_images_dir, class_name)
        test_class_dir = os.path.join(test_images_dir, class_name)

        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir, exist_ok=True)

        for img_file in os.listdir(valid_class_dir):
            src_file = os.path.join(valid_class_dir, img_file)
            dest_file = os.path.join(test_class_dir, img_file)
            shutil.move(src_file, dest_file)

    print("Moved all data from valid to test.")

if __name__ == "__main__":
    # Define source and output directories
    source_dir = "Tu_lam_v2"
    output_dir = "Tu_lam_v2_11"
    
    # Create the classification structure
    create_classification_structure(source_dir, output_dir)
    
    # Move valid data to test
    move_valid_to_test(source_dir, output_dir)
    
    print("Conversion and move completed successfully!") 