"""
Sinh ảnh thời tiết synthetic (rain/fog/snow) cho TRAIN SET để tăng đa dạng domain.
Yêu cầu: pip install albumentations opencv-python

Chạy:
  python scripts/augment_weather.py
"""

import os
from pathlib import Path
import random
import shutil
import cv2
import albumentations as A

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "data" / "traffic"
IN_DIR = ROOT / "images" / "train"
LBL_IN = ROOT / "labels" / "train"
OUT_DIR = ROOT / "images" / "train_weather"
LBL_OUT = ROOT / "labels" / "train_weather"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

augs = {
    "rain": A.Compose([A.RandomRain(blur_value=3, p=1.0)]),
    "fog":  A.Compose([A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.4, alpha_coef=0.1, p=1.0)]),
    "snow": A.Compose([A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=1.0, p=1.0)]),
}

def process_one(img_path: Path, weather: str):
    img = cv2.imread(str(img_path))
    if img is None: 
        return
    aug = augs[weather]
    out = aug(image=img)["image"]
    out_name = f"{img_path.stem}_{weather}{img_path.suffix}"
    cv2.imwrite(str(OUT_DIR / out_name), out)
    # Copy label 1:1
    lbl_src = LBL_IN / f"{img_path.stem}.txt"
    if lbl_src.exists():
        shutil.copy2(lbl_src, LBL_OUT / f"{img_path.stem}_{weather}.txt")

def main():
    imgs = [p for p in IN_DIR.glob("*.*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
    if not imgs:
        print(f"Không tìm thấy ảnh train trong {IN_DIR}. Hãy chuẩn bị dữ liệu trước.")
        return
    random.shuffle(imgs)
    # Tạo ~20% synthetic cho mỗi weather
    k = max(1, len(imgs)//5)
    for weather in ["rain","fog","snow"]:
        for p in imgs[:k]:
            process_one(p, weather)
    print("Done synthetic weather. Merge thư mục train_weather vào train nếu muốn oversample.")

if __name__ == "__main__":
    main()