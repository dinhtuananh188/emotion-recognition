import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO models
model1 = YOLO("runs/detect/train15/weights/best.pt")  # Model 1: Nhận diện khuôn mặt và cảm xúc
model2 = YOLO("runs/detect/train2/weights/best.pt")  # Model 2: So sánh nhãn cảm xúc

# Input and output directories
input_dir = "FEC_dataset/images"
output_dir = "FEC_dataset/output"
output_images_dir = os.path.join(output_dir, "images")
output_labels_dir = os.path.join(output_dir, "labels")

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)

    inter_area = max(0, xi2 - xi1, 0) * max(0, yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def merge_overlapping_boxes(boxes, threshold=0.8):
    """Merge overlapping bounding boxes."""
    merged_boxes = []
    for box in boxes:
        for mbox in merged_boxes:
            if iou(box[:4], mbox[:4]) > threshold:
                mbox[4].update(box[4])  # Merge labels
                break
        else:
            merged_boxes.append(box)
    return merged_boxes

def process_image(image_path):
    """Process a single image."""
    image = cv2.imread(image_path)
    
    # Kiểm tra nếu không thể đọc ảnh
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return  # Bỏ qua ảnh lỗi và tiếp tục
    
    h, w, _ = image.shape  # Kích thước ảnh
    results1 = model1.predict(image)[0]  
    boxes1 = []

    # Extract bounding boxes and labels from Model 1
    if results1.boxes is not None:
        for box in results1.boxes.data.tolist():  
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.2:  
                label_index = int(cls)  # Lấy chỉ số lớp thay vì tên nhãn
                boxes1.append([x1, y1, x2, y2, {label_index}])

    # Merge overlapping boxes
    merged_boxes = merge_overlapping_boxes(boxes1)

    # Validate with Model 2
    results2 = model2.predict(image)[0]  
    valid_faces = []
    if results2.boxes is not None:
        for box in merged_boxes:
            x1, y1, x2, y2, labels = box
            for result in results2.boxes.data.tolist():
                _, _, _, _, conf2, cls2 = result
                label2 = int(cls2)  
                if label2 in labels:
                    valid_faces.append([x1, y1, x2, y2, [(label2, conf2)]])
                    break

    # Save valid faces
    if valid_faces:
        base_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_images_dir, base_name)
        cv2.imwrite(output_image_path, image)  

        # Ensure label files are saved as .txt
        label_file_path = os.path.join(output_labels_dir, os.path.splitext(base_name)[0] + ".txt")
        with open(label_file_path, "w") as label_file:
            for box in valid_faces:
                x1, y1, x2, y2, labels = box
                x_center = ((x1 + x2) / 2) / w  
                y_center = ((y1 + y2) / 2) / h  
                width = (x2 - x1) / w  
                height = (y2 - y1) / h  

                # Handle multiple labels for the same bounding box
                if labels:
                    # Select the label with the highest confidence
                    label = max(labels, key=lambda lbl: lbl[1])[0]  # lbl[0] is the label, lbl[1] is the confidence
                    label_file.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    """Main function to process all images."""
    for image_name in os.listdir(input_dir):
        if image_name.endswith((".jpg", ".png")):
            process_image(os.path.join(input_dir, image_name))

if __name__ == "__main__":
    main()
