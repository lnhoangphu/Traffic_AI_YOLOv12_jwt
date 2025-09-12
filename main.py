from fastapi import FastAPI, File, UploadFile
from detect import detect_objects

app = FastAPI(
    title="Traffic AI Detection API",
    description="API for detecting objects in images using YOLOv8",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Traffic AI Detection API",
        "endpoints": {
            "/detect": "POST - Upload an image for object detection",
            "/docs": "Interactive API documentation"
        }
    }

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    objects = detect_objects(image_bytes)
    return {"objects": objects}