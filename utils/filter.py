import os

def is_label_invalid(label_path):
    """
    Kiểm tra xem file label có hợp lệ không.
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Kiểm tra file rỗng
        if not lines:
            return True

        for line in lines:
            parts = line.strip().split()

            # Kiểm tra định dạng: ít nhất phải có 5 phần tử (emotion_class, x, y, w, h)
            if len(parts) < 5:
                return True

            # Kiểm tra giá trị bounding box nằm trong khoảng hợp lệ [0,1]
            _, x_center, y_center, width, height = map(float, parts[:5])
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                return True

    except Exception as e:
        print(f"Lỗi khi đọc file {label_path}: {e}")
        return True  # Nếu có lỗi đọc file, coi như file bị hỏng

    return False

def delete_invalid_files():
    """
    Xóa các file ảnh và nhãn nếu nhãn bị lỗi.
    """
    input_dir = "Data/YOLO_format/train"
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("❌ Thư mục images hoặc labels không tồn tại!")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    deleted_count = 0
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path) and is_label_invalid(label_path):
            try:
                os.remove(image_path)  # Xóa ảnh
                os.remove(label_path)  # Xóa file nhãn
                deleted_count += 1
                print(f"🗑️ Đã xóa: {image_path} & {label_path}")
            except Exception as e:
                print(f"❌ Lỗi khi xóa {image_path} hoặc {label_path}: {e}")

    print(f"\n✅ Đã xóa {deleted_count} file ảnh và nhãn không hợp lệ.")

if __name__ == "__main__":
    delete_invalid_files()
