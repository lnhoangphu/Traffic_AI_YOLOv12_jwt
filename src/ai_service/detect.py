from ultralytics import YOLO
from PIL import Image
import io

model = YOLO('yolov8n.pt')  # Model nhẹ, tải tự động lần đầu

def infer(image_bytes):
    """
    Inference function for object detection using YOLOv8
    Args:
        image_bytes: Raw image bytes
    Returns:
        List of detected objects with label, confidence, and bounding box
    """
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img)
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

def detect_objects(image_bytes):
    """
    Legacy function name for compatibility
    """
    return infer(image_bytes)