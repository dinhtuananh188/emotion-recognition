import os
from tqdm import tqdm

def change_emotion_labels(labels_dir):
    """
    Change all emotion class labels to 9 in YOLO format files
    """
    if not os.path.exists(labels_dir):
        print(f"Thư mục labels không tồn tại: {labels_dir}")
        return

    # Get all .txt files in the labels directory
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"Tìm thấy {len(label_files)} file nhãn trong {labels_dir}")

    for label_file in tqdm(label_files, desc="Đang xử lý file nhãn"):
        label_path = os.path.join(labels_dir, label_file)
        
        # Read the current labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Modify the labels
        new_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Replace the emotion class with 9, keep the bounding box coordinates
                new_label = f"9 {' '.join(parts[1:])}\n"
                new_labels.append(new_label)
            else:
                # Keep invalid lines as is
                new_labels.append(line)
        
        # Write the modified labels back to file
        with open(label_path, 'w') as f:
            f.writelines(new_labels)

def main():
    print("=== Chương trình đổi nhãn cảm xúc sang số 9 ===")
    try:
        # Specify the directory containing the label files
        labels_dir = "Facial-Expression-3/train/labels"  # Adjust this path as needed
        change_emotion_labels(labels_dir)
        print("\nHoàn thành! Đã đổi tất cả nhãn cảm xúc sang số 9.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main() 