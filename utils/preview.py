import cv2
import os
import numpy as np

import cv2
import numpy as np

def resize_image(image, target_width=640):
    """
    Resize ảnh với chiều rộng cố định, chiều cao tự động điều chỉnh theo tỷ lệ
    
    Args:
        image: Ảnh input (numpy array)
        target_width: Chiều rộng cố định mong muốn (default: 640)
    Returns:  
        Ảnh đã được resize (numpy array)
    """
    height, width = image.shape[:2]
    
    # Tính toán tỷ lệ scale dựa trên chiều rộng
    scale = width / height
    
    # Tính chiều cao mới và ép kiểu về int
    target_height = int(target_width / scale)
    
    # Resize ảnh giữ nguyên tỷ lệ
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return resized


def draw_emotion_boxes(image_path, label_path):
    """
    Đọc ảnh và label file, vẽ bounding box và emotion
    """
    emotion_mapping = {
        0: "Anger",
        1: "Contempt", 
        2: "Disgust",
        3: "Fear",
        4: "Happy",
        5: "Neutral",
        6: "Sad",
        7: "Surprise"
    }
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Resize ảnh
    image = resize_image(image)
    height, width = image.shape[:2]
    
    # Đọc label file
    if not os.path.exists(label_path):
        print(f"Không tìm thấy file label: {label_path}. Tạo file label trống.")
        with open(label_path, 'w') as f:
            f.write("")
        return None
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        try:
            class_id, x_center, y_center, w, h = map(float, line.strip().split())
            
            # Chuyển từ format YOLO sang pixel coordinates
            x = int(x_center * width - (w * width) / 2)
            y = int(y_center * height - (h * height) / 2)
            w = int(w * width)
            h = int(h * height)

            
            # Vẽ bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Lấy tên emotion
            emotion = emotion_mapping.get(int(class_id), "Unknown")
            
            # Điều chỉnh kích thước text theo kích thước ảnh
            font_scale = min(width, height) * 0.001  # Scale font size based on image size
            thickness = max(1, int(min(width, height) * 0.002))  # Scale thickness
            
            # Thêm text hiển thị emotion
            text = f"{emotion}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Vẽ background cho text
            cv2.rectangle(image, 
                        (x, y - text_size[1] - 8), 
                        (x + text_size[0], y), 
                        (0, 255, 0), 
                        -1)
            
            # Vẽ text
            cv2.putText(image, 
                       text, 
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale,
                       (0, 0, 0),
                       thickness)
            
        except Exception as e:
            print(f"Lỗi khi xử lý label: {str(e)}")
            continue
    
    return image
def visualize_dataset():
    """
    Hiển thị tất cả ảnh trong dataset với labels, mỗi ảnh hiển thị trong một cửa sổ riêng
    """
    image_dir = "Facial-Expression-3/train/processed/Done/images"  
    label_dir = "Facial-Expression-3/train/processed/Done/labels"
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print("Không tìm thấy thư mục ảnh hoặc labels!")
        return
    
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    current_idx = 0
    
    while True:
        image_file = image_files[current_idx]
        image_path = os.path.join(image_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Vẽ bounding box và emotion
        result_image = draw_emotion_boxes(image_path, label_path)
        
        if result_image is not None:
            # Đóng cửa sổ cũ trước khi mở ảnh mới
            cv2.destroyAllWindows()
            
            # Hiển thị ảnh với một cửa sổ mới
            window_name = f'Image {current_idx + 1}/{len(image_files)}: {image_file}'
            cv2.imshow(window_name, result_image)
            
            # Xử lý phím bấm
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Thoát
                break
            elif key == ord('a') and current_idx > 0:  # Ảnh trước
                current_idx -= 1
            elif key == ord('d') and current_idx < len(image_files) - 1:  # Ảnh sau
                current_idx += 1
            elif key == ord('s'):  # Lưu ảnh
                save_path = os.path.join("Data/processed_data/visualized", image_file)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, result_image)
                print(f"Đã lưu ảnh tại: {save_path}")
    
    cv2.destroyAllWindows()

def main():
    print("=== Chương trình hiển thị ảnh với emotion labels ===")
    print("Controls:")
    print("- 'a': Ảnh trước")
    print("- 'd': Ảnh tiếp theo")
    print("- 's': Lưu ảnh")
    print("- 'q': Thoát")
    
    try:
        visualize_dataset()
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main()
