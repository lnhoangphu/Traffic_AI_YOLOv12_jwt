from ultralytics import YOLO
from PIL import Image
import io

# Nạp model YOLOv8 pretrained (có thể dùng yolov8n.pt cho nhẹ)
model = YOLO('yolov8n.pt')

def detect_objects(image_bytes):
    # Đọc ảnh từ bytes
    img = Image.open(io.BytesIO(image_bytes))
    # Nhận diện
    results = model(img)
    # Lấy thông tin bounding box, nhãn
    objects = []
    for box in results[0].boxes:
        label = results[0].names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        objects.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, x2, y2]
        })
    return objects