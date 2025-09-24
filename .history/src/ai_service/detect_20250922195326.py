from ultralytics import YOLO
from PIL import Image
import io
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np

# Sử dụng model đã train thay vì model gốc
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINED_MODEL = PROJECT_ROOT / "runs" / "balanced" / "balanced_training_20250922_1352252" / "weights" / "best.pt"

# Cấu hình từ .env
AUTO_USE_TRAINED_MODEL = os.getenv("AUTO_USE_TRAINED_MODEL", "true").lower() == "true"
CONFIDENCE_THRESHOLD = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("MODEL_IOU_THRESHOLD", "0.45"))
# Per-class thresholds and suppression rule (configurable via env)
PERSON_MIN_CONF = float(os.getenv("PERSON_MIN_CONF", "0.65"))
VEHICLE_MIN_CONF = float(os.getenv("VEHICLE_MIN_CONF", str(CONFIDENCE_THRESHOLD)))
SUPPRESS_PERSON_IF_IOU_WITH_VEHICLE = float(os.getenv("SUPPRESS_PERSON_IF_IOU_WITH_VEHICLE", "0.5"))

# Logic chọn model
if AUTO_USE_TRAINED_MODEL and TRAINED_MODEL.exists():
    MODEL_PATH = str(TRAINED_MODEL)
    print(f"🎯 Sử dụng model đã train: {MODEL_PATH}")
else:
    MODEL_PATH = os.getenv("YOLO_WEIGHTS", "yolo12n.pt")
    print(f"⚠️ Dùng model gốc: {MODEL_PATH}")

# Load model once
model = YOLO(MODEL_PATH)
# Warm-up
_ = model.predict(Image.new("RGB", (640, 640)), verbose=False)

def infer(image_bytes: bytes):
    """
    Phát hiện đối tượng trong ảnh
    Args:
        image_bytes: Dữ liệu ảnh dạng bytes
    Returns:
        List các đối tượng được phát hiện
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Sử dụng threshold từ .env
    results = model.predict(
        img, 
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )
    
    r = results[0]
    dets = _boxes_to_detections(r)
    dets = _postprocess_classwise_thresholds_and_overlap(dets, r)
    return dets

def _boxes_to_detections(r):
    dets = []
    if r.boxes is not None:
        for b in r.boxes:
            cls = int(b.cls)
            dets.append({
                "label": r.names[cls],
                "class_id": cls,
                "confidence": float(b.conf),
                "box_xyxy": [float(x) for x in b.xyxy[0].tolist()],
                "box_center": [
                    float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                    float((b.xyxy[0][1] + b.xyxy[0][3]) / 2)
                ],
                "box_size": [
                    float(b.xyxy[0][2] - b.xyxy[0][0]),
                    float(b.xyxy[0][3] - b.xyxy[0][1])
                ]
            })
    return dets

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _postprocess_classwise_thresholds_and_overlap(dets, r):
    # Apply per-class thresholds
    filtered = []
    for d in dets:
        label = d["label"].lower()
        conf = d["confidence"]
        if label == "person":
            if conf < PERSON_MIN_CONF:
                continue
        elif label in ("vehicle", "car"):
            if conf < VEHICLE_MIN_CONF:
                continue
        else:
            if conf < CONFIDENCE_THRESHOLD:
                continue
        filtered.append(d)

    # Suppress person if heavily overlapping a vehicle
    if SUPPRESS_PERSON_IF_IOU_WITH_VEHICLE > 0:
        vehicles = [d for d in filtered if d["label"].lower() in ("vehicle", "car")]
        kept = []
        for d in filtered:
            if d["label"].lower() == "person":
                suppress = False
                for v in vehicles:
                    iou = _iou_xyxy(d["box_xyxy"], v["box_xyxy"])
                    if iou >= SUPPRESS_PERSON_IF_IOU_WITH_VEHICLE and v["confidence"] >= (d["confidence"] - 0.1):
                        suppress = True
                        break
                if not suppress:
                    kept.append(d)
            else:
                kept.append(d)
        filtered = kept

    return filtered

def _predict_frame_detections(frame_bgr: np.ndarray):
    """Run detection on a single BGR frame and return list of detections and plotted frame (RGB)."""
    # Convert BGR to RGB for plotting consistency
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(
        frame_rgb,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )
    r = results[0]
    detections = _boxes_to_detections(r)
    detections = _postprocess_classwise_thresholds_and_overlap(detections, r)
    # r.plot() returns annotated image in RGB
    annotated_rgb = r.plot() if hasattr(r, "plot") else frame_rgb
    return detections, annotated_rgb

def analyze_video_to_json(
    video_bytes: bytes,
    sample_every: int = 1,
    max_frames: int | None = 300,
):
    """
    Analyze a video and return per-frame detections as JSON-friendly data.
    Only processes every `sample_every`-th frame up to `max_frames`.
    """
    if sample_every < 1:
        sample_every = 1

    # Write to a temporary file for OpenCV
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    cap = cv2.VideoCapture(tmp_in_path)
    if not cap.isOpened():
        os.remove(tmp_in_path)
        raise RuntimeError("Không thể mở video đầu vào")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    results = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % sample_every != 0:
                frame_index += 1
                continue

            dets, _ = _predict_frame_detections(frame)
            results.append({
                "frame_index": frame_index,
                "objects": dets
            })
            processed += 1
            frame_index += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        cap.release()
        os.remove(tmp_in_path)

    return {
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames
        },
        "sampling": {
            "sample_every": sample_every,
            "max_frames": max_frames
        },
        "frames": results
    }

def save_annotated_video(
    video_bytes: bytes,
    output_path: str,
    sample_every: int = 1,
    max_frames: int | None = None,
    output_fps: float | None = None,
):
    """
    Save an annotated MP4 video to `output_path`.
    """
    if sample_every < 1:
        sample_every = 1

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    cap = cv2.VideoCapture(tmp_in_path)
    if not cap.isOpened():
        os.remove(tmp_in_path)
        raise RuntimeError("Không thể mở video đầu vào")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = output_fps or in_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 640)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        os.remove(tmp_in_path)
        raise RuntimeError("Không thể tạo video đầu ra")

    frame_index = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % sample_every == 0:
                dets, annotated_rgb = _predict_frame_detections(frame)
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                writer.write(annotated_bgr)
                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break
            else:
                # If skipping processing, keep original frame to maintain timing
                writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        cap.release()
        os.remove(tmp_in_path)

    return {
        "fps": out_fps,
        "width": width,
        "height": height,
        "frames_written": processed
    }
