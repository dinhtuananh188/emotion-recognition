import os
import json

# filepath: d:\Downloads\Compressed\Train\FEC_dataset\processing_progress.json
# Đường dẫn đến file JSON
json_path = r'd:\Downloads\Compressed\Train\FEC_dataset\processing_progress.json'

# Đường dẫn đến thư mục chứa các folder cảm xúc
categorized_emotions_path = r'd:\Downloads\Compressed\Train\FEC_dataset\categorized_emotions'

# Đọc nội dung file JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Lấy danh sách các folder cảm xúc
emotion_folders = ['Anger', 'Contempt', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Tạo tập hợp chứa tất cả các ảnh trong các folder cảm xúc
valid_images = set()
for folder in emotion_folders:
    folder_path = os.path.join(categorized_emotions_path, folder, 'images')
    if os.path.exists(folder_path):
        valid_images.update(os.listdir(folder_path))

# Lọc lại JSON, chỉ giữ các ảnh có trong valid_images
filtered_data = {key: value for key, value in data.items() if key in valid_images}

# Ghi lại file JSON đã cập nhật
with open(json_path, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print("Đã cập nhật file JSON.")