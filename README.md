# Traffic AI YOLO (Baseline)

Mục tiêu:
- Phát hiện/phân loại: traffic_sign, motorcycle, pedestrian, car, truck, bicycle, pothole, bus.
- Hợp nhất nhiều dataset Kaggle về YOLO format thống nhất.
- Xử lý mất cân bằng dữ liệu và tích hợp yếu tố thời tiết (clear/rain/fog/snow).
- Cung cấp AI service (FastAPI) để frontend/back-end gọi.

Cấu trúc:
```
├─ src/
│  └─ ai_service/
│     ├─ main.py          # FastAPI server
│     └─ detect.py        # Wrapper YOLO
├─ scripts/
│  ├─ download_kaggle.ps1 # Tải dataset (Windows PowerShell)
│  ├─ download_kaggle.sh  # Tải dataset (Linux/Mac)
│  ├─ prepare_datasets.py # Hợp nhất & chuẩn hóa YOLO (skeleton, cần adapters)
│  └─ augment_weather.py  # Tạo ảnh thời tiết synthetic (rain/fog/snow)
├─ config/
│  └─ taxonomy.yaml       # Danh sách lớp mục tiêu + weather
├─ data/
│  └─ traffic/
│     └─ data.yaml        # Ultralytics data config (train/val/test + names)
├─ training/
│  └─ train.ps1           # Train baseline YOLO (Windows)
├─ .env.example           # Biến môi trường cho AI service
├─ requirements.txt       # Thư viện Python
└─ .gitignore
```

Cách dùng nhanh (Windows, PowerShell):
1) Tạo venv và cài dependency:
- python -m venv venv
- .\venv\Scripts\activate
- pip install -r requirements.txt

2) Đặt file kaggle.json ở thư mục gốc repo (cùng cấp README.md).
3) Tải datasets từ Kaggle:
- .\scripts\download_kaggle.ps1
  (Lưu tại datasets_src/..., tự unzip)

4) Chuẩn bị dataset hợp nhất (skeleton – cần bạn gửi cấu trúc để leader viết adapters):
- python .\scripts\prepare_datasets.py
  (Hiện tại script sẽ tạo khung thư mục và báo TODO cho từng dataset)

5) Train baseline YOLO:
- .\training\train.ps1

6) Chạy AI service:
- uvicorn src.ai_service.main:app --reload
- Mở http://localhost:8000/ để xem API info
- Mở http://localhost:8000/docs để test POST /detect
- Healthcheck: http://localhost:8000/healthz

Ghi chú:
- Nhiều dataset Kaggle KHÔNG ở YOLO format. Bạn hãy tải xong và gửi lại list thư mục + ví dụ file nhãn để leader viết adapters trong scripts/prepare_datasets.py.
- Khi chưa có dữ liệu unified, bạn vẫn có thể chạy AI API để test inference với model YOLO pre-trained.

Bảo mật/Hiệu suất:
- Bật REQUIRE_AUTH trong .env để yêu cầu JWT (khi tích hợp backend).
- Resize ảnh phía client trước khi upload để tăng tốc.
- CORS whitelist domain FE ở biến AI_ALLOWED_ORIGINS.