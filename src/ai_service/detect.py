from ultralytics import YOLO
from PIL import Image
import io
import os

MODEL_PATH = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")

# Load model once
model = YOLO(MODEL_PATH)
# Warm-up
_ = model.predict(Image.new("RGB", (640, 640)), verbose=False)

def infer(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(img, verbose=False)
    r = results[0]
    dets = []
    for b in r.boxes:
        cls = int(b.cls)
        dets.append({
            "label": r.names[cls],
            "class_id": cls,
            "confidence": float(b.conf),
            "box_xyxy": [float(x) for x in b.xyxy[0].tolist()]
        })
    return dets