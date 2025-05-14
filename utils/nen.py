import os
from PIL import Image

def compress_image(input_path, quality=85):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")  # Đảm bảo định dạng RGB để tránh lỗi PNG với transparency
            output_path = input_path  # Ghi đè ảnh gốc
            img.save(output_path, "JPEG", quality=quality)
            print(f"Đã nén: {input_path}")
    except Exception as e:
        print(f"Lỗi nén {input_path}: {e}")

def batch_compress_images(directory, size_threshold_mb=15, quality=85):
    size_threshold = size_threshold_mb * 1024 * 1024  # Chuyển đổi MB sang bytes
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if os.path.getsize(file_path) > size_threshold:
                    compress_image(file_path, quality)
            except Exception as e:
                print(f"Lỗi xử lý {file_path}: {e}")

if __name__ == "__main__":
    folder_path = "Data/Data_co_san/images"
    batch_compress_images(folder_path)
