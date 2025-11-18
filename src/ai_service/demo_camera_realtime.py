"""
Demo chạy YOLOv12 realtime với webcam/camera
Yêu cầu: opencv-python đã được cài (pip install opencv-python)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

# Load model đã train
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "train_11class_final" / "yolov12n_11class_weighted" / "weights" / "best.pt"

# Fallback nếu chưa có model trained
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "runs" / "train_balanced_final" / "yolov12n_11class_balanced" / "weights" / "best.pt"
    if not MODEL_PATH.exists():
        MODEL_PATH = PROJECT_ROOT / "runs" / "quick_train_11class" / "yolov12n_quick_test" / "weights" / "best.pt"
        if not MODEL_PATH.exists():
            MODEL_PATH = PROJECT_ROOT / "yolo12n.pt"
            print("⚠️  Chưa có model trained, dùng pretrained model")

print(f"📦 Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print("✅ Model loaded successfully!")

# Class names (11 classes)
CLASS_NAMES = {
    0: "Vehicle",
    1: "Bus", 
    2: "Bicycle",
    3: "Person",
    4: "Engine",
    5: "Truck",
    6: "Tricycle",
    7: "Obstacle",
    8: "Pothole",
    9: "Traffic Light",
    10: "Traffic Sign"
}

# Colors cho mỗi class (BGR format)
COLORS = {
    0: (0, 255, 0),      # Vehicle - Green
    1: (255, 0, 0),      # Bus - Blue
    2: (0, 255, 255),    # Bicycle - Yellow
    3: (255, 0, 255),    # Person - Magenta
    4: (128, 128, 0),    # Engine - Teal
    5: (0, 128, 255),    # Truck - Orange
    6: (255, 128, 0),    # Tricycle - Light Blue
    7: (0, 0, 255),      # Obstacle - Red
    8: (128, 0, 128),    # Pothole - Purple
    9: (0, 165, 255),    # Traffic Light - Orange
    10: (255, 255, 0)    # Traffic Sign - Cyan
}

def draw_detections(frame, results, conf_threshold=0.25):
    """
    Vẽ bounding boxes và labels lên frame
    """
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Lấy thông tin
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
                
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Lấy class name và color
            class_name = CLASS_NAMES.get(cls, f"Class {cls}")
            color = COLORS.get(cls, (255, 255, 255))
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label với background
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return frame


def main():
    """
    Chạy detection realtime từ webcam
    """
    print("\n" + "="*60)
    print("🎥 YOLOV12 REALTIME CAMERA DETECTION")
    print("="*60)
    print("\n📹 Camera options:")
    print("   0 = Webcam mặc định")
    print("   1 = Camera ngoài (nếu có)")
    print("   hoặc nhập đường dẫn IP camera: rtsp://...")
    
    # Chọn camera source
    camera_input = input("\n🎬 Chọn camera (Enter = 0): ").strip()
    
    if camera_input == "":
        camera_source = 0
    elif camera_input.isdigit():
        camera_source = int(camera_input)
    else:
        camera_source = camera_input  # IP camera or video file
    
    # Mở camera
    print(f"\n📡 Connecting to camera: {camera_source}...")
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        print("\n💡 Thử:")
        print("   1. Kiểm tra camera đã kết nối chưa")
        print("   2. Thử camera source khác (0, 1, 2...)")
        print("   3. Kiểm tra quyền truy cập camera")
        return
    
    # Lấy thông tin camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"✅ Camera connected!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print("\n" + "="*60)
    print("⌨️  CONTROLS:")
    print("   Q = Quit")
    print("   S = Save screenshot")
    print("   + = Increase confidence threshold")
    print("   - = Decrease confidence threshold")
    print("="*60 + "\n")
    
    # Biến tracking
    conf_threshold = 0.25
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không đọc được frame!")
                break
            
            frame_count += 1
            
            # Inference
            inference_start = time.time()
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Vẽ detections
            frame = draw_detections(frame, results, conf_threshold)
            
            # Tính FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_display = frame_count / elapsed
            
            # Đếm số objects detected
            total_detections = sum(len(r.boxes) for r in results)
            
            # Vẽ thông tin lên frame
            info_y = 30
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, info_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {total_detections}", (10, info_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf_threshold:.2f}", (10, info_y + 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị
            cv2.imshow("YOLOv12 Realtime Detection", frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Stopping...")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                screenshot_path = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"📈 Confidence threshold: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"📉 Confidence threshold: {conf_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("📊 SESSION STATS:")
        print(f"   Total frames: {frame_count}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Screenshots: {screenshot_count}")
        print("="*60)
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera closed. Goodbye!")


if __name__ == "__main__":
    main()
