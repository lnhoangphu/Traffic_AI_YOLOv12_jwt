from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from .detect import infer, model, MODEL_PATH, analyze_video_to_json, save_annotated_video
import os
import time
import psutil
import torch
from pathlib import Path

import cv2
import numpy as np
import tempfile
from typing import Optional

# Load environment variables với default values
API_TITLE = os.getenv("API_TITLE", "Traffic AI Service - YOLOv12")
API_VERSION = os.getenv("API_VERSION", "2.0.0")
TRAINING_DATASET_NAME = os.getenv("TRAINING_DATASET_NAME", "Traffic AI Balanced Dataset")

# Tạo app với thông tin chi tiết từ .env
app = FastAPI(
    title=API_TITLE,
    description=f"API phát hiện đối tượng giao thông sử dụng YOLOv12 đã được train trên {TRAINING_DATASET_NAME}",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("AI_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình từ .env
MAX_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024  # Convert MB to bytes
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.7"))
BENCHMARK_MAX_SAMPLES = int(os.getenv("BENCHMARK_MAX_SAMPLES", "50"))
DEFAULT_BENCHMARK_SAMPLES = int(os.getenv("DEFAULT_BENCHMARK_SAMPLES", "10"))
WARMUP_ITERATIONS = int(os.getenv("WARMUP_ITERATIONS", "3"))
SHOW_TRAINING_METRICS = os.getenv("SHOW_TRAINING_METRICS", "true").lower() == "true"
ENABLE_GPU_METRICS = os.getenv("ENABLE_GPU_METRICS", "true").lower() == "true"

def _auth(authorization: str | None):
    """Xác thực người dùng"""
    if REQUIRE_AUTH:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Thiếu hoặc sai token xác thực")
        # TODO: verify JWT signature (HS256/RS256) tùy backend của bạn

@app.get("/")
async def root():
    """Trang chủ API với danh sách endpoints"""
    return {
        "message": f"🚀 Chào mừng đến với {API_TITLE}",
        "model_info": f"YOLOv12 đã được train trên {TRAINING_DATASET_NAME}",
        "version": API_VERSION,
        "config": {
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "10")),
            "auth_required": REQUIRE_AUTH,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD
        },
        "endpoints": {
            "/detect": "POST - Tải lên ảnh để phát hiện đối tượng giao thông",
            "/model/info": "GET - Thông tin chi tiết về model",
            "/model/metrics": "GET - Hiệu suất hệ thống hiện tại", 
            "/test/benchmark": "POST - Đo tốc độ inference",
            "/healthz": "GET - Kiểm tra trạng thái API",
            "/docs": "📚 Tài liệu API tương tác (Swagger UI)",
            "/redoc": "📖 Tài liệu API (ReDoc)"
        },
        "classes": [
            "Vehicle", "Bus", "Bicycle", "Person", "Engine", 
            "Truck", "Tricycle", "Obstacle", "Pothole", 
            "Traffic Light", "Traffic Sign"
        ]
    }

@app.get("/healthz")
def healthz():
    """Kiểm tra trạng thái API"""
    return {
        "status": "ok", 
        "timestamp": time.time(),
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.get("/model/info")
async def model_info():
    """Thông tin chi tiết về model"""
    try:
        # Lấy training metrics từ .env nếu có
        performance_data = {}
        if SHOW_TRAINING_METRICS:
            performance_data = {
                "mAP@0.5": f"{os.getenv('TRAINING_MAP50', '62.8')}%",
                "mAP@0.5:0.95": f"{os.getenv('TRAINING_MAP50_95', '44.9')}%",
                "precision": f"{os.getenv('TRAINING_PRECISION', '79.6')}%",
                "recall": f"{os.getenv('TRAINING_RECALL', '55.7')}%"
            }
        
        return {
            "model_type": "YOLOv12n",
            "training_dataset": TRAINING_DATASET_NAME,
            "classes": list(model.names.values()) if hasattr(model, 'names') else [],
            "num_classes": len(model.names) if hasattr(model, 'names') else 0,
            "input_size": "640x640",
            "model_path": MODEL_PATH,
            "performance": performance_data,
            "thresholds": {
                "confidence": float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.25")),
                "iou": float(os.getenv("MODEL_IOU_THRESHOLD", "0.45")),
                "high_confidence": HIGH_CONFIDENCE_THRESHOLD,
                "person_min_conf": float(os.getenv("PERSON_MIN_CONF", "0.75")),
                "vehicle_min_conf": float(os.getenv("VEHICLE_MIN_CONF", "0.20")),
                "suppress_person_if_iou_with_vehicle": float(os.getenv("SUPPRESS_PERSON_IF_IOU_WITH_VEHICLE", "0.6"))
            },
            "training_info": {
                "epochs": 100,
                "batch_size": 4,
                "optimizer": "AdamW",
                "device": "CUDA (RTX 3050 Ti)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thông tin model: {str(e)}")

@app.get("/model/metrics")
async def model_metrics():
    """Thống kê hiệu suất hệ thống"""
    try:

    # API xử lý video: tách frame, detect từng frame, trả về kết quả JSON
        # Thông tin bộ nhớ
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Thông tin GPU nếu có và được bật
        gpu_info = {"gpu_available": False}
        if ENABLE_GPU_METRICS and torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 2),
                "gpu_memory_cached_mb": round(torch.cuda.memory_reserved(0) / 1e6, 2)
            }
        
        return {
            "system": {
                "ram_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": psutil.cpu_percent(),
                "cpu_count": psutil.cpu_count()
            },
            "gpu": gpu_info,
            "config": {
                "gpu_metrics_enabled": ENABLE_GPU_METRICS,
                "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "10")),
                "benchmark_max_samples": BENCHMARK_MAX_SAMPLES
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thông tin hệ thống: {str(e)}")

@app.post("/test/benchmark")
async def benchmark_inference(num_samples: int = DEFAULT_BENCHMARK_SAMPLES):
    """Đo tốc độ inference của model"""
    if num_samples > BENCHMARK_MAX_SAMPLES:
        raise HTTPException(
            status_code=400, 
            detail=f"Tối đa {BENCHMARK_MAX_SAMPLES} mẫu test (có thể thay đổi trong .env)"
        )
    
    try:
        import io
        import numpy as np
        from PIL import Image
        
        # Tạo ảnh test
        test_img = Image.new('RGB', (640, 640), color=(128, 128, 128))
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        test_data = img_bytes.getvalue()
        
        # Warm-up với số lần từ .env
        for _ in range(WARMUP_ITERATIONS):
            infer(test_data)
        
        # Đo tốc độ
        times = []
        for i in range(num_samples):
            start_time = time.time()
            results = infer(test_data)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # chuyển sang ms
        
        avg_time = np.mean(times)
        
        return {
            "test_info": {
                "num_samples": num_samples,
                "warmup_iterations": WARMUP_ITERATIONS,
                "image_size": "640x640",
                "model": f"YOLOv12n ({TRAINING_DATASET_NAME})"
            },
            "performance": {
                "avg_inference_time_ms": round(avg_time, 2),
                "min_time_ms": round(np.min(times), 2),
                "max_time_ms": round(np.max(times), 2),
                "std_time_ms": round(np.std(times), 2),
                "throughput_fps": round(1000 / avg_time, 2)
            },
            "status": "✅ Tốt" if avg_time < 100 else "⚠️ Chậm",
            "config": {
                "max_samples_allowed": BENCHMARK_MAX_SAMPLES,
                "default_samples": DEFAULT_BENCHMARK_SAMPLES
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi benchmark: {str(e)}")

@app.post("/video/detect")
async def video_detect(
    file: UploadFile = File(...),
    sample_every: int = 1,
    max_frames: Optional[int] = 300,
    authorization: str | None = Header(None)
):
    """
    Phát hiện đối tượng theo từng frame trong video và trả về JSON.
    - sample_every: lấy mẫu mỗi N frame (mặc định 1: mọi frame)
    - max_frames: giới hạn số frame xử lý để phản hồi nhanh
    """
    _auth(authorization)

    if file.content_type not in ("video/mp4", "video/quicktime", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ video MP4/MOV")

    data = await file.read()
    if len(data) > (MAX_SIZE * 5):  # Cho phép video lớn hơn ảnh một chút
        raise HTTPException(status_code=413, detail="Video quá lớn. Tối đa ~50MB (tùy cấu hình)")

    try:
        result = analyze_video_to_json(data, sample_every=sample_every, max_frames=max_frames)
        return {
            "success": True,
            "video": {
                "filename": file.filename,
                "size_bytes": len(data),
                "content_type": file.content_type
            },
            "model": {
                "name": "YOLOv12n Balanced",
                "conf": float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.25")),
                "iou": float(os.getenv("MODEL_IOU_THRESHOLD", "0.45"))
            },
            "result": result,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý video: {str(e)}")

@app.post("/video/stream")
async def video_stream(
    file: UploadFile = File(...),
    sample_every: int = 1,
    max_frames: Optional[int] = None,
    output_fps: Optional[float] = None,
    authorization: str | None = Header(None)
):
    """
    Trả về video MP4 đã được vẽ bounding boxes.
    """
    _auth(authorization)

    if file.content_type not in ("video/mp4", "video/quicktime", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ video MP4/MOV")

    data = await file.read()
    if len(data) > (MAX_SIZE * 10):
        raise HTTPException(status_code=413, detail="Video quá lớn. Tối đa ~100MB (tùy cấu hình)")

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
            out_path = tmp_out.name

        info = save_annotated_video(
            data,
            output_path=out_path,
            sample_every=sample_every,
            max_frames=max_frames,
            output_fps=output_fps
        )

        def iterfile():
            try:
                with open(out_path, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    os.remove(out_path)
                except Exception:
                    pass

        headers = {
            "X-Video-Width": str(info["width"]),
            "X-Video-Height": str(info["height"]),
            "X-Video-FPS": str(info["fps"]),
        }

        return StreamingResponse(iterfile(), media_type="video/mp4", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo video annotate: {str(e)}")

@app.post("/detect")
async def detect(file: UploadFile = File(...), authorization: str | None = Header(None)):
    """
    Phát hiện đối tượng giao thông trong ảnh
    
    - **file**: Ảnh cần phân tích (JPEG/PNG, tối đa 10MB)
    - **authorization**: Token xác thực (nếu bắt buộc)
    
    Trả về danh sách các đối tượng được phát hiện với tọa độ và độ tin cậy
    """
    _auth(authorization)
    
    # Kiểm tra định dạng file
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400, 
            detail="Chỉ hỗ trợ ảnh JPEG và PNG"
        )
    
    # Đọc file
    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File quá lớn. Tối đa {MAX_SIZE//1024//1024}MB"
        )
    
    try:
        # Thực hiện inference
        start_time = time.time()
        detections = infer(data)
        inference_time = (time.time() - start_time) * 1000
        
        # Thống kê kết quả
        class_counts = {}
        total_confidence = 0
        high_confidence_count = 0
        
        for det in detections:
            class_name = det["label"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += det["confidence"]
            if det["confidence"] > HIGH_CONFIDENCE_THRESHOLD:
                high_confidence_count += 1
        
        avg_confidence = total_confidence / len(detections) if detections else 0
        
        return {
            "success": True,
            "image_info": {
                "filename": file.filename,
                "size_bytes": len(data),
                "content_type": file.content_type
            },
            "inference": {
                "time_ms": round(inference_time, 2),
                "model": "YOLOv12n Balanced"
            },
            "results": {
                "total_objects": len(detections),
                "high_confidence_objects": high_confidence_count,
                "average_confidence": round(avg_confidence, 3),
                "class_distribution": class_counts,
                "objects": detections
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi xử lý ảnh: {str(e)}"
        )