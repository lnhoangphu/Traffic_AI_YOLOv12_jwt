#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Traffic AI Dataset
Quy trình tiền xử lý dữ liệu hoàn chỉnh trước khi training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import yaml
import shutil
from collections import Counter
import random
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, dataset_path, target_size=(640, 640)):
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        self.processed_path = self.dataset_path.parent / f"{self.dataset_path.name}_processed"
        self.splits = ['train', 'val', 'test']
        
    def create_processed_structure(self):
        """Tạo cấu trúc thư mục cho dữ liệu đã xử lý"""
        print("📁 Tạo cấu trúc thư mục processed...")
        
        # Remove existing processed folder
        if self.processed_path.exists():
            shutil.rmtree(self.processed_path)
        
        # Create new structure
        for split in self.splits:
            (self.processed_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.processed_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Copy data.yaml
        src_yaml = self.dataset_path / 'data.yaml'
        dst_yaml = self.processed_path / 'data.yaml'
        if src_yaml.exists():
            shutil.copy2(src_yaml, dst_yaml)
    
    def resize_image_with_padding(self, image, target_size):
        """Resize ảnh với padding để giữ aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale factor
        scale = min(target_w / w, target_h / h)
        
        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # Gray padding
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded, scale, pad_x, pad_y
    
    def adjust_bbox_coordinates(self, bbox_line, scale, pad_x, pad_y, original_size, target_size):
        """Điều chỉnh bbox coordinates sau resize và padding"""
        parts = bbox_line.strip().split()
        if len(parts) != 5:
            return None
        
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Convert to absolute coordinates
            orig_w, orig_h = original_size
            abs_x = x_center * orig_w
            abs_y = y_center * orig_h
            abs_w = width * orig_w
            abs_h = height * orig_h
            
            # Apply scale and padding
            new_x = abs_x * scale + pad_x
            new_y = abs_y * scale + pad_y
            new_w = abs_w * scale
            new_h = abs_h * scale
            
            # Convert back to relative coordinates
            target_w, target_h = target_size
            rel_x = new_x / target_w
            rel_y = new_y / target_h
            rel_w = new_w / target_w
            rel_h = new_h / target_h
            
            # Validate coordinates
            if (0 <= rel_x <= 1 and 0 <= rel_y <= 1 and 
                0 <= rel_w <= 1 and 0 <= rel_h <= 1 and
                rel_x - rel_w/2 >= 0 and rel_x + rel_w/2 <= 1 and
                rel_y - rel_h/2 >= 0 and rel_y + rel_h/2 <= 1):
                return f"{class_id} {rel_x:.6f} {rel_y:.6f} {rel_w:.6f} {rel_h:.6f}\n"
            else:
                return None
                
        except (ValueError, IndexError):
            return None
    
    def process_images_and_labels(self):
        """Xử lý ảnh và labels"""
        print(f"🖼️ Xử lý ảnh và labels (target size: {self.target_size})...")
        
        processed_count = 0
        error_count = 0
        
        for split in self.splits:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists():
                continue
            
            print(f"📂 Processing {split}...")
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for img_path in tqdm(image_files, desc=f"Processing {split}"):
                try:
                    # Read image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        error_count += 1
                        continue
                    
                    original_size = (image.shape[1], image.shape[0])  # (width, height)
                    
                    # Resize with padding
                    processed_img, scale, pad_x, pad_y = self.resize_image_with_padding(
                        image, self.target_size
                    )
                    
                    # Save processed image
                    output_img_path = self.processed_path / 'images' / split / img_path.name
                    cv2.imwrite(str(output_img_path), processed_img)
                    
                    # Process corresponding label
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    output_label_path = self.processed_path / 'labels' / split / f"{img_path.stem}.txt"
                    
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                        
                        processed_lines = []
                        for line in lines:
                            processed_line = self.adjust_bbox_coordinates(
                                line, scale, pad_x, pad_y, original_size, self.target_size
                            )
                            if processed_line:
                                processed_lines.append(processed_line)
                        
                        # Save processed labels
                        with open(output_label_path, 'w') as f:
                            f.writelines(processed_lines)
                    else:
                        # Create empty label file
                        output_label_path.touch()
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")
                    error_count += 1
        
        print(f"✅ Processed: {processed_count} images")
        print(f"❌ Errors: {error_count} images")
    
    def create_class_weights(self):
        """Tạo class weights để xử lý imbalance"""
        print("⚖️ Tính toán class weights...")
        
        class_counts = Counter()
        
        # Count classes in processed dataset
        labels_dir = self.processed_path / 'labels' / 'train'
        if not labels_dir.exists():
            return
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
                        except (ValueError, IndexError):
                            continue
        
        if not class_counts:
            return
        
        # Calculate weights (inverse frequency)
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = {}
        for class_id in sorted(class_counts.keys()):
            weight = total_samples / (num_classes * class_counts[class_id])
            weights[class_id] = weight
        
        # Save weights to file
        weights_file = self.processed_path / 'class_weights.yaml'
        with open(weights_file, 'w') as f:
            yaml.dump({'class_weights': weights}, f, default_flow_style=False)
        
        print(f"💾 Class weights saved to: {weights_file}")
        for class_id, weight in weights.items():
            print(f"   Class {class_id}: {weight:.4f}")
    
    def update_data_yaml(self):
        """Cập nhật data.yaml với đường dẫn mới"""
        data_yaml_path = self.processed_path / 'data.yaml'
        
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Update paths
            data['path'] = str(self.processed_path)
            data['train'] = 'images/train'
            data['val'] = 'images/val'
            data['test'] = 'images/test'
            
            # Add preprocessing info
            data['preprocessing'] = {
                'image_size': list(self.target_size),
                'padding_color': 114,
                'resize_method': 'letterbox'
            }
            
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            print(f"✅ Updated data.yaml: {data_yaml_path}")
    
    def run_preprocessing(self):
        """Chạy toàn bộ quy trình tiền xử lý"""
        print(f"🚀 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
        print(f"Dataset: {self.dataset_path.name}")
        print(f"Target size: {self.target_size}")
        print("=" * 60)
        
        # Create structure
        self.create_processed_structure()
        
        # Process images and labels
        self.process_images_and_labels()
        
        # Calculate class weights
        self.create_class_weights()
        
        # Update data.yaml
        self.update_data_yaml()
        
        print(f"\n🎉 HOÀN THÀNH TIỀN XỬ LÝ!")
        print(f"📁 Dữ liệu đã xử lý: {self.processed_path}")
        print("✅ Sẵn sàng cho training!")

def main():
    """Main function"""
    print("🔧 Data Preprocessing Pipeline")
    print("=" * 50)
    
    base_dir = Path(r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets")
    
    datasets = [
        "traffic_ai_balanced_11class",
        "traffic_ai_imbalanced_11class"
    ]
    
    target_size = (640, 640)  # YOLOv12 standard size
    
    for dataset_name in datasets:
        dataset_path = base_dir / dataset_name
        
        if dataset_path.exists():
            print(f"\n{'='*20} {dataset_name} {'='*20}")
            preprocessor = DataPreprocessor(dataset_path, target_size)
            preprocessor.run_preprocessing()
        else:
            print(f"❌ Dataset không tồn tại: {dataset_path}")

if __name__ == "__main__":
    main()