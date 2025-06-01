# Hệ Thống Nhận Diện Khuôn Mặt / Face Recognition System

## 📋 Yêu cầu / Requirements

### 🖥️ Yêu cầu hệ thống / System Requirements
- Python 3.10
- CUDA 12.4 (for GPU support)

### 📦 Thư viện Python / Python Dependencies
```bash
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 (Nếu muốn dùng GPU)
```

> ⚠️ **Lưu ý / Note**: Khuyến khích sử dụng GPU để có hiệu suất tốt hơn  
> **GPU is highly recommended for better performance**

## 🚀 Hướng dẫn sử dụng / How to Use

### Chạy ứng dụng web / Run Web Application
1. Mở terminal trong thư mục dự án / Open terminal in project folder
```bash
cd webapp
```

2. Chạy ứng dụng / Run application
```bash
python app.py
```

3. Truy cập ứng dụng / Access application
- Mở trình duyệt web / Open web browser
- Truy cập / Go to: `http://127.0.0.1:2402`

---

✅ Dự án đã sẵn sàng để phát triển và thử nghiệm hệ thống nhận diện khuôn mặt!
