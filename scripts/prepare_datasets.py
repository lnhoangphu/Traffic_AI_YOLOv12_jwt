"""
Hợp nhất datasets Kaggle về YOLO format thống nhất theo taxonomy.
Lưu ý: NHIỀU DATASET KHÔNG Ở YOLO FORMAT -> cần adapter chuyển đổi.
Trước mắt script tạo KHUNG thư mục và kiểm tra; leader sẽ bổ sung adapters
khi bạn gửi cấu trúc thư mục + ví dụ nhãn của từng bộ.

Chạy:
  python scripts/prepare_datasets.py
"""

import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_SRC = REPO_ROOT / "datasets_src"
OUT_ROOT = REPO_ROOT / "data" / "traffic"

# Mục tiêu đích (YOLO)
TARGET_DIRS = {
    "train_images": OUT_ROOT / "images" / "train",
    "val_images": OUT_ROOT / "images" / "val",
    "test_images": OUT_ROOT / "images" / "test",
    "train_labels": OUT_ROOT / "labels" / "train",
    "val_labels": OUT_ROOT / "labels" / "val",
    "test_labels": OUT_ROOT / "labels" / "test",
}

def ensure_dirs():
    for p in TARGET_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

def describe_source_tree():
    print("== Datasets source folders ==")
    if not DATASETS_SRC.exists():
        print("datasets_src/ chưa tồn tại. Hãy chạy scripts/download_kaggle trước.")
        return
    for p in sorted(DATASETS_SRC.iterdir()):
        if p.is_dir():
            print(f"- {p.name}/")
            # liệt kê một mức
            for c in sorted(p.iterdir()):
                print(f"  - {c.name}")

def main():
    print(f"Đang kiểm tra từ thư mục gốc: {REPO_ROOT}")
    print(f"Thư mục datasets_src: {DATASETS_SRC}")
    print("\n=== Bắt đầu chuẩn bị dữ liệu ===\n")
    
    ensure_dirs()
    describe_source_tree()

    print("\n== TODO adapters ==")
    print("Mỗi dataset cần một adapter convert annotation -> YOLO:")
    print("1) road_issues            -> map về class 6: pothole (và các defect khác nếu có).")
    print("2) object_detection_35    -> chọn person/car/truck/bus/bicycle/motorbike -> map taxonomy.")
    print("3) vn_traffic_sign        -> gộp mọi loại biển báo -> class 0: traffic_sign.")
    print("4) intersection_flow_5k   -> dùng làm nền (copy-paste) hoặc bổ trợ nếu có bbox.")
    print("\nHãy chạy script này sau khi tải datasets; GỬI ẢNH CHỤP cấu trúc thư mục và 1-2 file nhãn điển hình")
    print("cho leader, tôi sẽ viết adapter cụ thể và hợp nhất dữ liệu.")
    print("\nKhi hợp nhất xong, OUT sẽ có:")
    print(str(OUT_ROOT / 'images/train') + " và " + str(OUT_ROOT / 'labels/train') + " ...")
    print("Sau đó bạn có thể train bằng training/train.ps1")

if __name__ == "__main__":
    main()