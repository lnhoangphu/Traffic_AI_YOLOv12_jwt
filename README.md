# Traffic AI Detection API

API nhận diện đối tượng giao thông sử dụng YOLOv8 / Traffic object detection API using YOLOv8

## Yêu cầu hệ thống / System Requirements

- Python 3.8 hoặc phiên bản mới hơn / Python 3.8 or newer
- pip (Python package installer)

## Cài đặt / Installation

1. Clone dự án / Clone the project:
```bash
git clone <repository-url>
cd Traffic_AI_YOLOv12_jwt
```

2. Tạo môi trường ảo / Create virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Cài đặt các thư viện / Install dependencies:
```bash
pip install -r requirements.txt
```

## Chạy ứng dụng / Run the application

1. Khởi động server / Start the server:
```bash
uvicorn main:app --reload
```

Server sẽ chạy tại / The server will run at: `http://127.0.0.1:8000`

## Các API Endpoints

### 1. Root Endpoint (GET `/`)
- Hiển thị thông tin về API và các endpoints có sẵn
- Shows API information and available endpoints

### 2. Object Detection (POST `/detect`)
- Upload ảnh để nhận diện đối tượng
- Upload an image for object detection

#### Cách sử dụng / How to use:
```python
import requests

# Gửi ảnh để nhận diện / Send image for detection
with open('your_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://127.0.0.1:8000/detect', files=files)

print(response.json())
```

### Documentation

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Cấu trúc dự án / Project Structure

```
Traffic_AI_YOLOv12_jwt/
├── main.py             # FastAPI application
├── detect.py          # YOLOv8 detection logic
├── requirements.txt   # Project dependencies
├── yolov8n.pt        # YOLOv8 model (downloaded automatically)
└── README.md         # This file
```

## Kết quả trả về / Response Format

```json
{
    "objects": [
        {
            "label": "car",
            "confidence": 0.95,
            "box": [100, 200, 300, 400]
        },
        // ... other detected objects
    ]
}
```

- `label`: Tên đối tượng được nhận diện / Name of detected object
- `confidence`: Độ tin cậy (0-1) / Confidence score (0-1)
- `box`: Tọa độ khung chứa [x1, y1, x2, y2] / Bounding box coordinates [x1, y1, x2, y2]