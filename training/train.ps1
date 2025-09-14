# Train baseline YOLOv8n trên dữ liệu đã hợp nhất
# Yêu cầu: data/traffic/data.yaml đã đúng và có ảnh/nhãn trong images/labels train/val/test
$ErrorActionPreference = "Stop"

$DATA_CFG = "data/traffic/data.yaml"
if (-not (Test-Path $DATA_CFG)) {
  Write-Error "Không tìm thấy $DATA_CFG. Hãy chuẩn bị dữ liệu trước."
}

# epochs/imgsz có thể chỉnh
yolo task=detect mode=train model=yolov8n.pt data=$DATA_CFG epochs=100 imgsz=640 patience=20