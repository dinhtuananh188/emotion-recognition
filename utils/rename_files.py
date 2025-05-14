import os
from tqdm import tqdm

def rename_files(images_dir, labels_dir):
    """
    Rename both image and label files to sequential numbers
    """
    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"Thư mục ảnh không tồn tại: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"Thư mục labels không tồn tại: {labels_dir}")
        return

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort files to ensure consistent ordering
    
    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    label_files.sort()

    print(f"Tìm thấy {len(image_files)} ảnh và {len(label_files)} file nhãn")

    # Rename image files
    for idx, old_name in enumerate(tqdm(image_files, desc="Đang đổi tên ảnh"), 1):
        # Get file extension
        ext = os.path.splitext(old_name)[1]
        # Create new name with leading zeros (e.g., 001.jpg)
        new_name = f"{idx:04d}{ext}"
        
        old_path = os.path.join(images_dir, old_name)
        new_path = os.path.join(images_dir, new_name)
        
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            print(f"Lỗi khi đổi tên file {old_name}: {str(e)}")

    # Rename label files
    for idx, old_name in enumerate(tqdm(label_files, desc="Đang đổi tên nhãn"), 1):
        new_name = f"{idx:04d}.txt"
        
        old_path = os.path.join(labels_dir, old_name)
        new_path = os.path.join(labels_dir, new_name)
        
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            print(f"Lỗi khi đổi tên file {old_name}: {str(e)}")

def main():
    print("=== Chương trình đổi tên file ảnh và nhãn ===")
    try:
        # Specify the directories containing the image and label files
        images_dir = "FEC_dataset/images"  # Adjust this path as needed
        labels_dir = "FEC_dataset/labels"  # Adjust this path as needed
        
        rename_files(images_dir, labels_dir)
        print("\nHoàn thành! Đã đổi tên tất cả file thành công.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main() 