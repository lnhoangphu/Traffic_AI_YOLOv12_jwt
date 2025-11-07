# 📹 YOLOv12 Realtime Camera Detection

## 📦 Thư viện cần thiết

Script sử dụng **OpenCV (cv2)** để xử lý camera realtime.

### ✅ Cài đặt (đã có sẵn)
```bash
pip install opencv-python
```

## 🚀 Cách sử dụng

### 1. Chạy demo camera
```bash
python demo_camera_realtime.py
```

### 2. Chọn nguồn camera
Khi chạy, script sẽ hỏi bạn chọn camera:
- **0** = Webcam mặc định (laptop webcam)
- **1** = Camera USB ngoài
- **rtsp://...** = IP Camera (ví dụ: camera giám sát)
- **video.mp4** = File video

### 3. Phím điều khiển
Khi demo đang chạy:
- **Q** = Thoát
- **S** = Chụp màn hình (screenshot)
- **+** = Tăng confidence threshold (+0.05)
- **-** = Giảm confidence threshold (-0.05)

## 📊 Thông tin hiển thị

Trên màn hình sẽ hiển thị:
- **FPS**: Số khung hình/giây
- **Inference**: Thời gian xử lý mỗi frame (ms)
- **Detections**: Số đối tượng phát hiện được
- **Conf**: Ngưỡng confidence hiện tại

## 🎯 11 Classes được phát hiện

| ID | Class | Màu sắc |
|----|-------|---------|
| 0 | Vehicle | Xanh lá |
| 1 | Bus | Xanh dương |
| 2 | Bicycle | Vàng |
| 3 | Person | Hồng |
| 4 | Engine | Teal |
| 5 | Truck | Cam |
| 6 | Tricycle | Xanh nhạt |
| 7 | Obstacle | Đỏ |
| 8 | Pothole | Tím |
| 9 | Traffic Light | Cam đậm |
| 10 | Traffic Sign | Cyan |

## 💡 Tips

### Tối ưu hiệu năng
- **GPU**: Script tự động dùng GPU nếu có (RTX 3050 Ti)
- **FPS cao**: Giảm resolution camera nếu cần
- **Confidence**: Điều chỉnh để giảm false positives

### Sử dụng IP Camera
```python
# Ví dụ RTSP stream
camera_source = "rtsp://username:password@192.168.1.100:554/stream"
```

### Xử lý video file
```python
# Chạy detection trên video file
camera_source = "traffic_video.mp4"
```

## 🔧 Tùy chỉnh nâng cao

### Thay đổi confidence threshold mặc định
Sửa trong `demo_camera_realtime.py`:
```python
conf_threshold = 0.25  # Mặc định
conf_threshold = 0.5   # Nghiêm ngặt hơn
conf_threshold = 0.15  # Dễ dàng hơn
```

### Thay đổi resolution
```python
# Thêm sau khi mở camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Lưu video output
```python
# Thêm video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Trong loop
out.write(frame)  # Sau khi vẽ detections

# Cleanup
out.release()
```

## 🐛 Troubleshooting

### Camera không mở được
```
❌ Không thể mở camera!
```
**Giải pháp:**
1. Kiểm tra camera đã kết nối
2. Thử camera source khác (0, 1, 2...)
3. Kiểm tra quyền truy cập camera (Settings → Privacy → Camera)
4. Đóng các app khác đang dùng camera (Zoom, Teams...)

### FPS thấp
**Nguyên nhân:**
- CPU/GPU yếu
- Resolution camera cao
- Quá nhiều objects trong frame

**Giải pháp:**
- Giảm resolution camera
- Tăng confidence threshold (ít detections hơn)
- Đảm bảo dùng GPU (check `check_gpu.py`)

### Bounding box nhấp nháy
- Điều chỉnh confidence threshold
- Sử dụng tracking algorithm (DeepSORT, ByteTrack)

## 📈 Benchmark

### RTX 3050 Ti (4GB)
- Resolution: 640x640
- Batch size: 1
- FPS: ~30-60 (tùy số objects)
- Inference: 6-10ms/frame

### CPU Only (i5-12500H)
- Resolution: 640x640
- FPS: ~5-10
- Inference: 50-100ms/frame

## 🔗 Tích hợp với ứng dụng khác

### Flask/FastAPI
```python
from demo_camera_realtime import model

# Trong API endpoint
results = model.predict(frame, conf=0.25)
```

### WebRTC
- Sử dụng `aiortc` để stream qua web
- Tham khảo: `src/ai_service/main.py` (video endpoints)

### Mobile App
- Export model sang TFLite/ONNX
- Deploy lên Edge device

## 🎓 Học thêm

### OpenCV Documentation
- [Camera Capture](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [Video I/O](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)

### YOLO Documentation
- [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/)
- [Real-time Detection](https://docs.ultralytics.com/guides/streamlit-live-inference/)

## 📝 Next Steps

1. **Test ngay**: `python demo_camera_realtime.py`
2. **Điều chỉnh**: Thử các confidence threshold
3. **Benchmark**: Kiểm tra FPS trên máy của bạn
4. **Tích hợp**: Thêm vào ứng dụng của bạn

---

**💡 TIP**: Chạy script với camera smartphone qua IP Webcam app để test với camera chất lượng cao hơn!
