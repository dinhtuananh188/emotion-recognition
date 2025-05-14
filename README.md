# Hệ Thống Nhận Diện Khuôn Mặt / Face Recognition System

## Yêu cầu / Requirements

### Yêu cầu hệ thống / System Requirements
- Python 3.10
- CUDA 11.2 (Bắt buộc nếu muốn train bằng GPU Nvidia / Required for GPU Nvidia training)
- cuDNN 8.1 (Bắt buộc nếu muốn train bằng GPU Nvidia / Required for GPU Nvidia training)

### Các thư viện Python cần thiết / Python Dependencies

#### Cho máy hỗ trợ GPU / For GPU Support:
```bash
pip install deepface
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow==2.10
pip install tenorflow_gpu=2.10.0
```

#### Cho máy chỉ dùng CPU / For CPU Only:
```bash
pip install deepface
pip install ultralytics
pip install torch torchvision torchaudio
```

Lưu ý / Note: Khuyến khích sử dụng GPU để có hiệu suất tốt hơn / GPU is highly recommended for better performance.

## Cài đặt / Installation

1. Clone repository này / Clone this repository
2. Cài đặt các thư viện cần thiết dựa trên cấu hình hệ thống của bạn / Install the required dependencies based on your system configuration
3. Đảm bảo bạn có phiên bản CUDA và cuDNN phù hợp nếu sử dụng GPU / Make sure you have the correct CUDA and cuDNN versions if using GPU