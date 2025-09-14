from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from .detect import infer  # Use relative import
import os
app = FastAPI(title="Traffic AI Service")
app = FastAPI(
    title="Traffic AI Service",
    description="API phát hiện đối tượng giao thông sử dụng YOLOv8",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("AI_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIZE = 10 * 1024 * 1024  # 10MB
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

def _auth(authorization: str | None):
    if REQUIRE_AUTH:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        # TODO: verify JWT signature (HS256/RS256) tùy backend của bạn
# Thêm route gốc "/"
@app.get("/")
async def root():
    return {
        "message": "Chào mừng đến với Traffic AI Service",
        "endpoints": {
            "/detect": "POST - Tải lên ảnh để phát hiện đối tượng",
            "/healthz": "GET - Kiểm tra trạng thái API",
            "/docs": "Tài liệu API tương tác (Swagger UI)",
            "/redoc": "Tài liệu API (ReDoc)"
        }
    }
@app.get("/healthz")
# Kiểm tra trạng thái API
def healthz():
    return {"status": "ok"}
# Route để xử lý phát hiện đối tượng
@app.post("/detect")
async def detect(file: UploadFile = File(...), authorization: str | None = Header(None)):
    _auth(authorization)
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Unsupported content type")
    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    dets = infer(data)
    return {"objects": dets}