"""
Object Detection 35 Classes Organizer - KEEP ORIGINAL 35 CLASSES
Tổ chức lại Object Detection 35 dataset giữ nguyên 35 classes như mô tả chính thức
Chỉ clean up structure và tạo class names đúng, KHÔNG convert sang 11 classes

Input: datasets_src/object_detection_35/final batches/ (35 classes)
Output: datasets_src/object_detection_35_organized/ (35 classes with correct names)

Usage: python scripts/organize_object_detection_35_keep_original.py
"""

import os
import sys
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

class ObjectDetection35Organizer:
    def __init__(self):
        self.input_root = Path("datasets_src/object_detection_35/final batches")
        self.output_root = Path("datasets_src/object_detection_35_organized")
        
        # OFFICIAL 35 class names theo mô tả chính thức từ Kaggle
        self.official_class_names = [
            "Person",           # 0
            "Chair",            # 1
            "Toothbrush",       # 2
            "Knife",            # 3
            "Bottle",           # 4
            "Cup",              # 5
            "Spoon",            # 6
            "Bench",            # 7
            "Refrigerator",     # 8
            "Fork",             # 9
            "Bus",              # 10
            "Toilet",           # 11
            "Bicycle",          # 12
            "Airplane",         # 13
            "Truck",            # 14
            "Motorcycles",      # 15
            "Oven",             # 16
            "Dog",              # 17
            "Bed",              # 18
            "Cat",              # 19
            "Traffic Light",    # 20 ⭐ Traffic-related
            "Currency",         # 21
            "Face",             # 22
            "Stop Sign",        # 23 ⭐ Traffic-related
            "Car",              # 24 ⭐ Traffic-related
            "Barriers",         # 25 ⭐ Traffic-related
            "Path Holes",       # 26 ⭐ Traffic-related
            "Stairs",           # 27
            "Train",            # 28 ⭐ Traffic-related
            "Bin",              # 29
            "Blind Stick",      # 30
            "Men Sign",         # 31
            "Cell Phone",       # 32
            "Women Sign",       # 33
            "Tap"               # 34
        ]
        
        # Train/Val/Test split ratios
        self.split_ratios = {'train': 0.8, 'val': 0.15, 'test': 0.05}
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': Counter(),
            'split_distribution': Counter()
        }
    
    def collect_all_files(self):
        """Thu thập tất cả image-label pairs từ các batches"""
        print("📸 Collecting all image-label pairs from batches...")
        
        all_pairs = []
        
        # Process each batch
        for batch_dir in self.input_root.iterdir():
            if batch_dir.is_dir() and batch_dir.name.startswith("Batch"):
                print(f"   📂 Processing {batch_dir.name}...")
                
                images_dir = batch_dir / "images"
                labels_dir = batch_dir / "labels"
                
                if not images_dir.exists() or not labels_dir.exists():
                    continue
                
                # Process each split in batch
                for split_dir in images_dir.iterdir():
                    if split_dir.is_dir():
                        split_labels = labels_dir / split_dir.name
                        if split_labels.exists():
                            
                            image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
                            
                            for img_file in image_files:
                                label_file = split_labels / f"{img_file.stem}.txt"
                                if label_file.exists():
                                    all_pairs.append({
                                        'image_path': img_file,
                                        'label_path': label_file,
                                        'batch': batch_dir.name,
                                        'original_split': split_dir.name
                                    })
        
        print(f"📊 Found {len(all_pairs)} image-label pairs")
        
        # Analyze class distribution
        for pair in all_pairs:
            try:
                with open(pair['label_path'], 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            self.stats['class_distribution'][class_id] += 1
                            self.stats['total_annotations'] += 1
            except:
                continue
        
        self.stats['total_images'] = len(all_pairs)
        
        return all_pairs
    
    def create_new_splits(self, all_pairs):
        """Tạo train/val/test splits mới với tỷ lệ chuẩn"""
        print(f"\\n🔄 Creating new train/val/test splits...")
        
        # Shuffle để random distribution
        random.seed(42)
        random.shuffle(all_pairs)
        
        # Calculate split indices
        n_total = len(all_pairs)
        n_train = int(n_total * self.split_ratios['train'])
        n_val = int(n_total * self.split_ratios['val'])
        
        # Assign new splits
        train_pairs = all_pairs[:n_train]
        val_pairs = all_pairs[n_train:n_train + n_val]
        test_pairs = all_pairs[n_train + n_val:]
        
        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        # Print split info
        for split_name, pairs in splits.items():
            count = len(pairs)
            percentage = count / n_total * 100
            print(f"   {split_name:5s}: {count:5,d} images ({percentage:4.1f}%)")
            self.stats['split_distribution'][split_name] = count
        
        return splits
    
    def copy_files_to_split(self, pairs, split_name):
        """Copy images và labels cho một split"""
        print(f"   📁 Copying {split_name} files...")
        
        # Create directories
        img_dir = self.output_root / 'images' / split_name
        label_dir = self.output_root / 'labels' / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for i, pair in enumerate(tqdm(pairs, desc=f"   {split_name}")):
            # Create new filename với prefix
            batch_name = pair['batch'].lower().replace(' ', '_')
            new_name = f"od35_{batch_name}_{i:06d}"
            
            # Copy image
            img_dest = img_dir / f"{new_name}.jpg"
            shutil.copy2(pair['image_path'], img_dest)
            
            # Copy label
            label_dest = label_dir / f"{new_name}.txt"
            shutil.copy2(pair['label_path'], label_dest)
    
    def create_dataset_config(self):
        """Tạo YAML config với 35 classes đầy đủ"""
        print("\\n📝 Creating dataset configuration...")
        
        # Create data.yaml with ORIGINAL 35 classes
        yaml_content = {
            'path': str(self.output_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 35,  # KEEP 35 classes
            'names': self.official_class_names
        }
        
        yaml_path = self.output_root / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Created: {yaml_path}")
        
        # Save statistics
        stats_content = {
            'dataset_name': 'Object Detection 35 Classes (VisionGuard)',
            'total_classes': 35,
            'total_images': self.stats['total_images'],
            'total_annotations': self.stats['total_annotations'],
            'split_distribution': dict(self.stats['split_distribution']),
            'class_distribution': dict(self.stats['class_distribution']),
            'class_names': self.official_class_names,
            'traffic_related_classes': {
                0: "Person",
                10: "Bus", 
                12: "Bicycle",
                14: "Truck",
                15: "Motorcycles",
                20: "Traffic Light",
                23: "Stop Sign", 
                24: "Car",
                25: "Barriers",
                26: "Path Holes",
                28: "Train"
            }
        }
        
        stats_path = self.output_root / "dataset_info.yaml"
        with open(stats_path, 'w', encoding='utf-8') as f:
            yaml.dump(stats_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Created: {stats_path}")
    
    def print_summary(self):
        """In tổng kết dataset"""
        print("\\n📊 DATASET ORGANIZATION SUMMARY")
        print("=" * 60)
        
        print(f"📂 Input: {self.input_root}")
        print(f"📂 Output: {self.output_root}")
        print(f"📸 Total images: {self.stats['total_images']:,}")
        print(f"📝 Total annotations: {self.stats['total_annotations']:,}")
        print(f"🏷️ Classes: 35 (ORIGINAL)")
        
        # Split distribution
        print(f"\\n📊 SPLIT DISTRIBUTION:")
        for split, count in self.stats['split_distribution'].items():
            percentage = count / self.stats['total_images'] * 100
            print(f"   {split:5s}: {count:5,d} images ({percentage:4.1f}%)")
        
        # Top classes
        print(f"\\n🔝 TOP 10 CLASSES:")
        top_classes = self.stats['class_distribution'].most_common(10)
        for class_id, count in top_classes:
            class_name = self.official_class_names[class_id] if class_id < len(self.official_class_names) else f"Class_{class_id}"
            percentage = count / self.stats['total_annotations'] * 100
            traffic_mark = "🚦" if class_id in [0, 10, 12, 14, 15, 20, 23, 24, 25, 26, 28] else "  "
            print(f"   {class_id:2d} {class_name:15s}: {count:5,d} ({percentage:4.1f}%) {traffic_mark}")
        
        # Traffic-related summary
        traffic_classes = [0, 10, 12, 14, 15, 20, 23, 24, 25, 26, 28]
        traffic_annotations = sum(self.stats['class_distribution'].get(cid, 0) for cid in traffic_classes)
        traffic_percentage = traffic_annotations / self.stats['total_annotations'] * 100
        
        print(f"\\n🚦 TRAFFIC-RELATED CLASSES: {len(traffic_classes)} classes")
        print(f"   Annotations: {traffic_annotations:,} ({traffic_percentage:.1f}%)")
    
    def organize(self):
        """Main organization process"""
        print("🎯 OBJECT DETECTION 35 CLASSES ORGANIZER - KEEP ORIGINAL")
        print("=" * 70)
        print("📋 Goal: Organize dataset structure while keeping all 35 classes")
        
        if not self.input_root.exists():
            print(f"❌ Input dataset not found: {self.input_root}")
            return
        
        # Clean output directory
        if self.output_root.exists():
            print("🗑️ Cleaning existing output...")
            shutil.rmtree(self.output_root)
        
        # Collect all files
        all_pairs = self.collect_all_files()
        
        if not all_pairs:
            print("❌ No image-label pairs found!")
            return
        
        # Create new splits
        splits = self.create_new_splits(all_pairs)
        
        # Copy files to new structure
        print("\\n📦 Copying files to new structure...")
        for split_name, pairs in splits.items():
            self.copy_files_to_split(pairs, split_name)
        
        # Create configuration
        self.create_dataset_config()
        
        # Print summary
        self.print_summary()
        
        print(f"\\n✅ ORGANIZATION COMPLETE!")
        print(f"📁 Output: {self.output_root}")
        print(f"🏷️ Classes: 35 (preserved original)")
        print(f"🎯 Ready for merge with other datasets!")

if __name__ == "__main__":
    organizer = ObjectDetection35Organizer()
    organizer.organize()