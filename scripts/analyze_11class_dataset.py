"""
Script phân tích dataset 11 class để kiểm tra:
- Số lượng samples mỗi class
- Distribution giữa train/val/test
- Validate rằng chỉ có đúng 11 class (0-10)
"""

import os
from pathlib import Path
from collections import Counter, defaultdict

# Định nghĩa 11 class chuẩn
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

def analyze_label_file(label_path):
    """Đọc file label và trả về list class_id"""
    class_ids = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_ids.append(class_id)
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
    return class_ids

def analyze_dataset(dataset_root):
    """Phân tích toàn bộ dataset"""
    
    dataset_path = Path(dataset_root)
    
    # Kiểm tra cấu trúc thư mục
    splits = ['train', 'val', 'test']
    
    results = {
        'total_images': {},
        'total_objects': {},
        'class_distribution': {},
        'invalid_classes': set(),
        'missing_labels': []
    }
    
    print("=" * 80)
    print(f"PHÂN TÍCH DATASET: {dataset_root}")
    print("=" * 80)
    
    for split in splits:
        labels_dir = dataset_path / 'labels' / split
        images_dir = dataset_path / 'images' / split
        
        if not labels_dir.exists():
            print(f"\n⚠️  Không tìm thấy: {labels_dir}")
            continue
            
        # Đếm số lượng images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        results['total_images'][split] = len(image_files)
        
        print(f"\n📁 {split.upper()}:")
        print(f"   Images: {len(image_files)}")
        print(f"   Labels: {len(label_files)}")
        
        # Phân tích class distribution
        class_counter = Counter()
        total_objects = 0
        
        for label_file in label_files:
            class_ids = analyze_label_file(label_file)
            class_counter.update(class_ids)
            total_objects += len(class_ids)
            
            # Kiểm tra class không hợp lệ
            for cid in class_ids:
                if cid not in CLASS_NAMES:
                    results['invalid_classes'].add(cid)
        
        results['total_objects'][split] = total_objects
        results['class_distribution'][split] = dict(class_counter)
        
        # Kiểm tra missing labels
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                results['missing_labels'].append(str(img_file))
        
        print(f"   Total objects: {total_objects}")
        print(f"\n   Class distribution:")
        for class_id in sorted(CLASS_NAMES.keys()):
            count = class_counter.get(class_id, 0)
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"      {class_id}: {CLASS_NAMES[class_id]:15s} - {count:5d} ({percentage:5.2f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT:")
    print("=" * 80)
    
    total_imgs = sum(results['total_images'].values())
    total_objs = sum(results['total_objects'].values())
    
    print(f"\n✅ Tổng số images: {total_imgs}")
    print(f"   - Train: {results['total_images'].get('train', 0)} ({results['total_images'].get('train', 0)/total_imgs*100:.1f}%)")
    print(f"   - Val:   {results['total_images'].get('val', 0)} ({results['total_images'].get('val', 0)/total_imgs*100:.1f}%)")
    print(f"   - Test:  {results['total_images'].get('test', 0)} ({results['total_images'].get('test', 0)/total_imgs*100:.1f}%)")
    
    print(f"\n✅ Tổng số objects: {total_objs}")
    
    # Tổng hợp class distribution
    total_class_dist = Counter()
    for split in splits:
        if split in results['class_distribution']:
            total_class_dist.update(results['class_distribution'][split])
    
    print(f"\n✅ Class distribution (toàn dataset):")
    for class_id in sorted(CLASS_NAMES.keys()):
        count = total_class_dist.get(class_id, 0)
        percentage = (count / total_objs * 100) if total_objs > 0 else 0
        print(f"   {class_id}: {CLASS_NAMES[class_id]:15s} - {count:6d} ({percentage:5.2f}%)")
    
    # Warnings
    if results['invalid_classes']:
        print(f"\n⚠️  CLASS KHÔNG HỢP LỆ PHÁT HIỆN: {sorted(results['invalid_classes'])}")
        print("   ❌ Dataset có class ngoài 0-10! Cần làm sạch lại!")
    else:
        print(f"\n✅ Tất cả class đều hợp lệ (0-10)")
    
    if results['missing_labels']:
        print(f"\n⚠️  Có {len(results['missing_labels'])} images thiếu label file")
    else:
        print(f"\n✅ Không có images thiếu label")
    
    # Kiểm tra class balance
    print(f"\n📊 Đánh giá cân bằng class:")
    if total_objs > 0:
        avg_count = total_objs / len(CLASS_NAMES)
        for class_id in sorted(CLASS_NAMES.keys()):
            count = total_class_dist.get(class_id, 0)
            ratio = count / avg_count if avg_count > 0 else 0
            status = "⚠️ " if ratio < 0.3 or ratio > 3.0 else "✅"
            print(f"   {status} {CLASS_NAMES[class_id]:15s}: ratio = {ratio:.2f}x average")
    
    print("\n" + "=" * 80)
    
    return results

if __name__ == "__main__":
    # Phân tích balanced dataset
    balanced_path = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets\traffic_ai_balanced_11class_processed"
    
    if os.path.exists(balanced_path):
        results = analyze_dataset(balanced_path)
        
        # Lưu kết quả
        output_file = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\dataset_11class_analysis.txt"
        print(f"\n💾 Kết quả đã được lưu vào: {output_file}")
    else:
        print(f"❌ Không tìm thấy dataset: {balanced_path}")
