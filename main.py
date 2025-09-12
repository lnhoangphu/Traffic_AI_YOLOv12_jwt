from fastapi import FastAPI, File, UploadFile
from detect import detect_objects

app = FastAPI()

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    objects = detect_objects(image_bytes)
    return {"objects": objects}