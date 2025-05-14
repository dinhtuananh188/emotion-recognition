import cv2
import os
import mediapipe as mp
from tqdm import tqdm

def detect_faces(image_path):
    """
    Detect faces in image using MediaPipe and return bounding boxes
    """
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return None, []

    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    bounding_boxes = []
    if results.detections:
        ih, iw, _ = image.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            # Convert to YOLO format (x_center, y_center, width, height)
            x_center = (x + w / 2) / iw
            y_center = (y + h / 2) / ih
            width = w / iw
            height = h / ih

            bounding_boxes.append((x_center, y_center, width, height))

    # Giải phóng bộ nhớ MediaPipe
    face_detection.close()

    return image, bounding_boxes

def update_labels(image_path, label_path):
    """
    Update the bounding box coordinates in the label file
    """
    # Detect faces
    _, face_boxes = detect_faces(image_path)
    if not face_boxes:
        print(f"Không tìm thấy khuôn mặt trong ảnh: {image_path}")
        return

    # Đọc file nhãn cũ
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Cập nhật nhãn với tọa độ mới
    new_labels = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        emotion_class = parts[0]  # Giữ nguyên emotion_class
        if i < len(face_boxes):
            x_center, y_center, width, height = face_boxes[i]
            new_labels.append(f"{emotion_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        else:
            new_labels.append(line)  # Giữ nguyên các nhãn không có khuôn mặt tương ứng

    # Ghi nhãn mới vào file
    with open(label_path, 'w') as f:
        f.writelines(new_labels)

def process_data_directory():
    """
    Process all images and labels in the data directory
    """
    input_dir = "FEC_dataset"
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Thư mục images hoặc labels không tồn tại!")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Tìm thấy {len(image_files)} ảnh trong {images_dir}")

    for image_file in tqdm(image_files, desc="Đang xử lý ảnh"):
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Không tìm thấy file nhãn cho ảnh: {image_file}")
            continue

        update_labels(image_path, label_path)

        # Giải phóng bộ nhớ ảnh
        del image_path
        del label_path

    print("\nHoàn thành! Đã cập nhật nhãn cho tất cả ảnh.")

def main():
    print("=== Chương trình cập nhật nhãn khuôn mặt ===")
    try:
        process_data_directory()
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main()
