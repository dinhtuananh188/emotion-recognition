import os
import shutil
import logging
from collections import Counter, defaultdict
import random

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Bản đồ cảm xúc sang chỉ số lớp
emotion_to_class = {
    'angry': 0, 'fear': 1, 'disgust': 2,
    'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
}
class_to_emotion = {v: k for k, v in emotion_to_class.items()}

# Giới hạn số nhãn cho mỗi cảm xúc
MAX_LABELS_PER_EMOTION = 70
# Giới hạn số khuôn mặt trong một ảnh
MAX_FACES_PER_IMAGE = 3

# Đường dẫn
input_folder = "/mnt/d/Downloads/Compressed/Train/FEC_dataset/processed"
output_folder = "/mnt/d/Downloads/Compressed/Train/FEC_dataset/balanced_dataset"

input_images_folder = os.path.join(input_folder, "images")
input_labels_folder = os.path.join(input_folder, "labels")

output_images_folder = os.path.join(output_folder, "images")
output_labels_folder = os.path.join(output_folder, "labels")

os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

def count_faces_and_emotions(label_path):
    """Đếm số khuôn mặt và lấy danh sách cảm xúc từ file nhãn"""
    emotions_count = Counter()
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_idx = int(parts[0])
                if class_idx in class_to_emotion:
                    emotions_count[class_to_emotion[class_idx]] += 1
        
        return len(lines), emotions_count
    except:
        return 0, emotions_count

def main():
    logging.info("Bắt đầu lọc và cân bằng dữ liệu cảm xúc...")
    
    # Lưu trữ thông tin về các ảnh và số lượng nhãn mỗi cảm xúc
    image_data = []
    
    # Dict để theo dõi số nhãn đã có trong mỗi cảm xúc
    emotion_label_counts = Counter()
    
    # Danh sách tất cả các file nhãn
    all_label_files = [f for f in os.listdir(input_labels_folder) if f.endswith('.txt')]
    logging.info(f"Tìm thấy {len(all_label_files)} file nhãn để xử lý")
    
    # Đi qua tất cả các file nhãn để phân loại
    for label_file in all_label_files:
        base_name = os.path.splitext(label_file)[0]
        image_file = None
        
        # Tìm file ảnh tương ứng
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            if os.path.exists(os.path.join(input_images_folder, base_name + ext)):
                image_file = base_name + ext
                break
        
        if not image_file:
            logging.warning(f"Không tìm thấy ảnh cho nhãn: {label_file}")
            continue
        
        label_path = os.path.join(input_labels_folder, label_file)
        
        # Đếm số khuôn mặt và cảm xúc
        face_count, emotions_count = count_faces_and_emotions(label_path)
        
        # Bỏ qua ảnh có quá nhiều khuôn mặt
        if face_count > MAX_FACES_PER_IMAGE:
            logging.info(f"Bỏ qua {image_file}: có {face_count} khuôn mặt (> {MAX_FACES_PER_IMAGE})")
            continue
        
        if not emotions_count:
            logging.warning(f"Không tìm thấy cảm xúc hợp lệ trong {label_file}")
            continue
        
        # Thêm thông tin ảnh vào danh sách
        image_data.append({
            'image': image_file,
            'label': label_file,
            'face_count': face_count,
            'emotions': emotions_count
        })
    
    # Xáo trộn danh sách ảnh để đa dạng hóa dữ liệu
    random.shuffle(image_data)
    
    # Dict để lưu các ảnh đã chọn
    selected_images = set()
    
    # Lọc ảnh để có đủ số nhãn cho mỗi cảm xúc
    for item in image_data:
        # Kiểm tra xem ảnh này có thể thêm vào tập dữ liệu không
        can_add = False
        
        for emotion, count in item['emotions'].items():
            if emotion_label_counts[emotion] < MAX_LABELS_PER_EMOTION:
                can_add = True
                break
        
        if can_add:
            # Thêm ảnh này vào tập dữ liệu
            selected_images.add(item['image'])
            
            # Cập nhật số lượng nhãn
            for emotion, count in item['emotions'].items():
                # Chỉ thêm số nhãn tới mức tối đa
                available_slots = MAX_LABELS_PER_EMOTION - emotion_label_counts[emotion]
                if available_slots > 0:
                    added_count = min(count, available_slots)
                    emotion_label_counts[emotion] += added_count
    
    # Sao chép các ảnh và nhãn được chọn
    for item in image_data:
        if item['image'] in selected_images:
            shutil.copy2(
                os.path.join(input_images_folder, item['image']),
                os.path.join(output_images_folder, item['image'])
            )
            
            shutil.copy2(
                os.path.join(input_labels_folder, item['label']),  
                os.path.join(output_labels_folder, item['label'])
            )
    
    logging.info("\nKết quả cuối cùng - số nhãn theo cảm xúc:")
    for emotion, count in emotion_label_counts.items():
        logging.info(f"{emotion}: {count} nhãn")
    
    logging.info(f"Tổng số ảnh trong tập dữ liệu cân bằng: {len(selected_images)}")
    logging.info(f"Tổng số nhãn: {sum(emotion_label_counts.values())}")

if __name__ == "__main__":
    main()