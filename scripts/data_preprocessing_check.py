#!/usr/bin/env python3
"""
Comprehensive Data Preprocessing Check for Traffic AI Dataset
Kiểm tra các vấn đề tiền xử lý dữ liệu trước khi huấn luyện
"""

import os
import cv2
import numpy as np
from pathlib import Path
import yaml
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessingChecker:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.splits = ['train', 'val', 'test']
        self.issues = []
        self.stats = {}
        
    def load_data_yaml(self):
        """Load data.yaml configuration"""
        data_yaml = self.dataset_path / 'data.yaml'
        if data_yaml.exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return None
    
    def check_image_sizes(self):
        """1. Kiểm tra kích thước ảnh và tính nhất quán"""
        print("🔍 1. KIỂM TRA KÍCH THƯỚC ẢNH")
        print("=" * 50)
        
        sizes = []
        corrupted_images = []
        size_distribution = Counter()
        
        for split in self.splits:
            images_dir = self.dataset_path / 'images' / split
            if not images_dir.exists():
                continue
                
            print(f"📂 Checking {split} images...")
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        corrupted_images.append(str(img_path))
                        continue
                    
                    h, w = img.shape[:2]
                    sizes.append((w, h))
                    size_distribution[(w, h)] += 1
                    
                except Exception as e:
                    corrupted_images.append(f"{img_path}: {str(e)}")
        
        if sizes:
            sizes_array = np.array(sizes)
            unique_sizes = len(size_distribution)
            most_common_size = size_distribution.most_common(1)[0]
            
            print(f"📊 Tổng số ảnh: {len(sizes)}")
            print(f"📏 Kích thước trung bình: {sizes_array.mean(axis=0).astype(int)}")
            print(f"📏 Kích thước phổ biến nhất: {most_common_size[0]} (xuất hiện {most_common_size[1]} lần)")
            print(f"🔢 Số loại kích thước khác nhau: {unique_sizes}")
            
            if unique_sizes > 10:
                self.issues.append(f"❌ Kích thước ảnh không đồng nhất ({unique_sizes} loại khác nhau)")
            else:
                print("✅ Kích thước ảnh tương đối đồng nhất")
        
        if corrupted_images:
            print(f"🚨 Phát hiện {len(corrupted_images)} ảnh lỗi:")
            for img in corrupted_images[:5]:  # Show first 5
                print(f"   - {img}")
            if len(corrupted_images) > 5:
                print(f"   ... và {len(corrupted_images) - 5} ảnh khác")
            self.issues.append(f"❌ {len(corrupted_images)} ảnh bị lỗi")
        
        self.stats['total_images'] = len(sizes)
        self.stats['corrupted_images'] = len(corrupted_images)
        self.stats['size_variations'] = unique_sizes
        
    def check_annotations(self):
        """2. Kiểm tra annotation YOLO format"""
        print("\n🔍 2. KIỂM TRA ANNOTATION YOLO")
        print("=" * 50)
        
        missing_labels = []
        invalid_labels = []
        class_distribution = Counter()
        bbox_issues = []
        
        data_config = self.load_data_yaml()
        num_classes = data_config.get('nc', 0) if data_config else 0
        
        for split in self.splits:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
                
            print(f"📂 Checking {split} annotations...")
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for img_path in image_files:
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                # Check if label file exists
                if not label_path.exists():
                    missing_labels.append(str(img_path))
                    continue
                
                # Check label content
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if len(parts) != 5:
                            invalid_labels.append(f"{label_path}:{line_num} - Wrong format")
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Check class ID range
                            if class_id < 0 or (num_classes > 0 and class_id >= num_classes):
                                invalid_labels.append(f"{label_path}:{line_num} - Invalid class ID: {class_id}")
                            
                            # Check bbox coordinates (should be 0-1)
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 <= width <= 1 and 0 <= height <= 1):
                                bbox_issues.append(f"{label_path}:{line_num} - Bbox out of range")
                            
                            class_distribution[class_id] += 1
                            
                        except ValueError:
                            invalid_labels.append(f"{label_path}:{line_num} - Invalid numbers")
                            
                except Exception as e:
                    invalid_labels.append(f"{label_path}: {str(e)}")
        
        print(f"📊 Phân bố classes:")
        if data_config and 'names' in data_config:
            for class_id in sorted(class_distribution.keys()):
                class_name = data_config['names'][class_id] if class_id < len(data_config['names']) else f"Class_{class_id}"
                print(f"   {class_id}: {class_name} - {class_distribution[class_id]} objects")
        
        if missing_labels:
            print(f"🚨 {len(missing_labels)} ảnh thiếu annotation")
            self.issues.append(f"❌ {len(missing_labels)} ảnh thiếu annotation")
        
        if invalid_labels:
            print(f"🚨 {len(invalid_labels)} annotation lỗi format")
            for issue in invalid_labels[:5]:
                print(f"   - {issue}")
            if len(invalid_labels) > 5:
                print(f"   ... và {len(invalid_labels) - 5} lỗi khác")
            self.issues.append(f"❌ {len(invalid_labels)} annotation lỗi format")
        
        if bbox_issues:
            print(f"🚨 {len(bbox_issues)} bbox coordinates lỗi")
            self.issues.append(f"❌ {len(bbox_issues)} bbox coordinates lỗi")
        
        self.stats['missing_labels'] = len(missing_labels)
        self.stats['invalid_labels'] = len(invalid_labels)
        self.stats['bbox_issues'] = len(bbox_issues)
        self.stats['class_distribution'] = dict(class_distribution)
        
    def check_data_balance(self):
        """3. Kiểm tra cân bằng dữ liệu"""
        print("\n🔍 3. PHÂN TÍCH CÂN BẰNG DỮ LIỆU")
        print("=" * 50)
        
        if 'class_distribution' not in self.stats:
            print("❌ Không thể phân tích do thiếu thông tin class distribution")
            return
        
        class_counts = list(self.stats['class_distribution'].values())
        if not class_counts:
            print("❌ Không có dữ liệu class nào")
            return
        
        min_count = min(class_counts)
        max_count = max(class_counts)
        mean_count = np.mean(class_counts)
        std_count = np.std(class_counts)
        cv = std_count / mean_count if mean_count > 0 else 0
        
        print(f"📊 Số lượng object:")
        print(f"   - Ít nhất: {min_count}")
        print(f"   - Nhiều nhất: {max_count}")
        print(f"   - Trung bình: {mean_count:.1f}")
        print(f"   - Tỷ lệ imbalance: {max_count/min_count:.2f}:1")
        print(f"   - Coefficient of Variation: {cv:.3f}")
        
        if cv > 0.5:
            self.issues.append(f"❌ Dữ liệu mất cân bằng nghiêm trọng (CV = {cv:.3f})")
            print(f"🚨 Dữ liệu mất cân bằng nghiêm trọng (CV = {cv:.3f})")
        elif cv > 0.3:
            self.issues.append(f"⚠️ Dữ liệu hơi mất cân bằng (CV = {cv:.3f})")
            print(f"⚠️ Dữ liệu hơi mất cân bằng (CV = {cv:.3f})")
        else:
            print(f"✅ Dữ liệu tương đối cân bằng (CV = {cv:.3f})")
        
        self.stats['imbalance_ratio'] = max_count/min_count if min_count > 0 else float('inf')
        self.stats['cv'] = cv
    
    def check_augmentation_potential(self):
        """4. Đánh giá khả năng augmentation"""
        print("\n🔍 4. ĐÁNH GIÁ AUGMENTATION")
        print("=" * 50)
        
        # Sample some images to check diversity
        train_images_dir = self.dataset_path / 'images' / 'train'
        if not train_images_dir.exists():
            print("❌ Không tìm thấy thư mục train images")
            return
        
        image_files = list(train_images_dir.glob('*.jpg'))[:50]  # Sample 50 images
        if not image_files:
            print("❌ Không tìm thấy ảnh training")
            return
        
        brightness_values = []
        contrast_values = []
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                contrast = np.std(gray)
                
                brightness_values.append(brightness)
                contrast_values.append(contrast)
                
            except Exception:
                continue
        
        if brightness_values:
            brightness_std = np.std(brightness_values)
            contrast_std = np.std(contrast_values)
            
            print(f"📊 Đa dạng về brightness: {brightness_std:.2f}")
            print(f"📊 Đa dạng về contrast: {contrast_std:.2f}")
            
            print(f"💡 Các kỹ thuật augmentation được khuyến nghị:")
            print(f"   - Random rotation (±10-15°)")
            print(f"   - Random scaling (0.8-1.2)")
            print(f"   - Brightness adjustment (±20%)")
            print(f"   - Random horizontal flip")
            print(f"   - Color jittering")
            print(f"   - Random crop/zoom")
        
    def generate_recommendations(self):
        """5. Đưa ra khuyến nghị cải thiện"""
        print("\n🎯 5. KHUYẾN NGHỊ TIỀN XỬ LÝ")
        print("=" * 50)
        
        if not self.issues:
            print("✅ Dữ liệu đã sẵn sàng cho training!")
            return
        
        print("🔧 Các vấn đề cần khắc phục:")
        for issue in self.issues:
            print(f"   {issue}")
        
        print("\n💡 Khuyến nghị cụ thể:")
        
        # Image size recommendations
        if self.stats.get('size_variations', 0) > 10:
            print("📏 Chuẩn hóa kích thước ảnh:")
            print("   - Resize tất cả ảnh về 640x640 (hoặc 512x512)")
            print("   - Sử dụng padding để giữ aspect ratio")
        
        # Corruption recommendations
        if self.stats.get('corrupted_images', 0) > 0:
            print("🔧 Xử lý ảnh lỗi:")
            print("   - Loại bỏ hoặc thay thế ảnh bị corrupted")
            print("   - Kiểm tra lại quá trình download/convert")
        
        # Annotation recommendations
        if self.stats.get('missing_labels', 0) > 0:
            print("📝 Xử lý annotation thiếu:")
            print("   - Loại bỏ ảnh không có label")
            print("   - Hoặc tạo label mới cho những ảnh quan trọng")
        
        # Balance recommendations
        if self.stats.get('cv', 0) > 0.5:
            print("⚖️ Cân bằng dữ liệu:")
            print("   - Oversampling cho classes ít")
            print("   - Data augmentation đặc biệt cho classes thiếu")
            print("   - Class weighting trong loss function")
            print("   - Focal loss để xử lý imbalance")
    
    def run_full_check(self):
        """Chạy tất cả các kiểm tra"""
        print(f"🔍 KIỂM TRA TIỀN XỬ LÝ DỮ LIỆU")
        print(f"Dataset: {self.dataset_path.name}")
        print("=" * 60)
        
        self.check_image_sizes()
        self.check_annotations()
        self.check_data_balance()
        self.check_augmentation_potential()
        self.generate_recommendations()
        
        print(f"\n📋 TÓM TẮT KIỂM TRA")
        print("=" * 60)
        print(f"📊 Tổng số vấn đề: {len(self.issues)}")
        
        if not self.issues:
            print("🎉 Dữ liệu đã sẵn sàng cho training!")
        else:
            print("⚠️ Cần xử lý các vấn đề trước khi training.")

def main():
    """Main function"""
    base_dir = Path(r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets")
    
    datasets = {
        "Balanced": base_dir / "traffic_ai_balanced_11class",
        "Imbalanced": base_dir / "traffic_ai_imbalanced_11class"
    }
    
    for name, dataset_path in datasets.items():
        if dataset_path.exists():
            print(f"\n{'='*20} {name} Dataset {'='*20}")
            checker = DataPreprocessingChecker(dataset_path)
            checker.run_full_check()
        else:
            print(f"❌ Dataset không tồn tại: {dataset_path}")

if __name__ == "__main__":
    main()