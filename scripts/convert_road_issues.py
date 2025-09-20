"""
Road Issues Dataset Converter - Classification to YOLO Format
Chuyển đổi Road Issues dataset từ classification folders sang YOLO format

Input: datasets_src/road_issues/data/ (folders by class)
Output: datasets_src/road_issues_yolo/ (YOLO format)

Usage: python scripts/convert_road_issues.py
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml

class RoadIssuesConverter:
    def __init__(self):
        self.input_root = Path("datasets_src/road_issues/data")
        self.output_root = Path("datasets_src/road_issues_yolo")
        
        # Mapping từ tên thư mục sang class index và tên chuẩn
        self.class_mapping = {
            "Broken Road Sign Issues": {"index": 0, "name": "broken_road_sign"},
            "Damaged Road issues": {"index": 1, "name": "damaged_road"}, 
            "Illegal Parking Issues": {"index": 2, "name": "illegal_parking"},
            "Mixed Issues": {"index": 3, "name": "mixed_issue"},
            "Pothole Issues": {"index": 4, "name": "pothole"},
            "Littering Garbage on Public Places Issues": {"index": 5, "name": "littering_garbage"},
            "Vandalism Issues": {"index": 6, "name": "vandalism"}
        }
        
        # Train/Val split ratio
        self.train_ratio = 0.8
        
    def discover_classes(self):
        """Khám phá các class từ thư mục"""
        print("🔍 Discovering classes from folder structure...")
        
        discovered_classes = []
        
        # Tìm trong Public Cleanliness + Environmental Issues
        public_env_path = self.input_root / "Public Cleanliness + Environmental Issues"
        if public_env_path.exists():
            for subdir in public_env_path.iterdir():
                if subdir.is_dir():
                    discovered_classes.append(subdir.name)
        
        # Tìm trong Road Issues
        road_issues_path = self.input_root / "Road Issues"
        if road_issues_path.exists():
            for subdir in road_issues_path.iterdir():
                if subdir.is_dir():
                    discovered_classes.append(subdir.name)
        
        print(f"📁 Found {len(discovered_classes)} classes:")
        for i, class_name in enumerate(discovered_classes):
            mapped_info = self.class_mapping.get(class_name, {"index": i, "name": class_name.lower().replace(" ", "_")})
            print(f"   {mapped_info['index']}: {class_name} -> {mapped_info['name']}")
        
        return discovered_classes
    
    def collect_images(self):
        """Thu thập tất cả ảnh từ các thư mục class"""
        print("\n📸 Collecting images from class folders...")
        
        all_images = []
        class_counts = Counter()
        
        # Collect from Public Cleanliness + Environmental Issues
        public_env_path = self.input_root / "Public Cleanliness + Environmental Issues"
        if public_env_path.exists():
            for class_dir in public_env_path.iterdir():
                if class_dir.is_dir() and class_dir.name in self.class_mapping:
                    class_info = self.class_mapping[class_dir.name]
                    
                    for img_file in class_dir.glob("*.jpg"):
                        all_images.append({
                            "path": img_file,
                            "class_index": class_info["index"],
                            "class_name": class_info["name"]
                        })
                        class_counts[class_info["name"]] += 1
        
        # Collect from Road Issues
        road_issues_path = self.input_root / "Road Issues"
        if road_issues_path.exists():
            for class_dir in road_issues_path.iterdir():
                if class_dir.is_dir() and class_dir.name in self.class_mapping:
                    class_info = self.class_mapping[class_dir.name]
                    
                    for img_file in class_dir.glob("*.jpg"):
                        all_images.append({
                            "path": img_file,
                            "class_index": class_info["index"],
                            "class_name": class_info["name"]
                        })
                        class_counts[class_info["name"]] += 1
        
        print(f"📊 Total images: {len(all_images)}")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} images")
        
        return all_images, class_counts
    
    def create_yolo_annotation(self, class_index):
        """Tạo YOLO annotation cho toàn bộ ảnh (classification -> detection)"""
        # Với classification, ta coi toàn bộ ảnh là 1 object
        # YOLO format: class_id center_x center_y width height (normalized 0-1)
        return f"{class_index} 0.5 0.5 1.0 1.0"
    
    def split_and_convert(self, all_images):
        """Chia train/val và chuyển đổi sang YOLO format"""
        print(f"\n🔄 Converting to YOLO format (train: {self.train_ratio:.0%}, val: {1-self.train_ratio:.0%})...")
        
        # Shuffle images
        random.shuffle(all_images)
        
        # Split train/val
        split_idx = int(len(all_images) * self.train_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        print(f"   Train: {len(train_images)} images")
        print(f"   Val: {len(val_images)} images")
        
        # Create output directories
        for split in ['train', 'val']:
            (self.output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Process train set
        for i, img_data in enumerate(train_images):
            self.copy_image_and_label(img_data, 'train', i)
        
        # Process val set
        for i, img_data in enumerate(val_images):
            self.copy_image_and_label(img_data, 'val', i)
    
    def copy_image_and_label(self, img_data, split, index):
        """Copy image và tạo label file"""
        img_path = img_data["path"]
        class_index = img_data["class_index"]
        class_name = img_data["class_name"]
        
        # New filename: road_issues_classname_index.jpg
        new_name = f"road_issues_{class_name}_{index:06d}"
        
        # Copy image
        img_dest = self.output_root / 'images' / split / f"{new_name}.jpg"
        shutil.copy2(img_path, img_dest)
        
        # Create YOLO label
        label_dest = self.output_root / 'labels' / split / f"{new_name}.txt"
        yolo_annotation = self.create_yolo_annotation(class_index)
        
        with open(label_dest, 'w') as f:
            f.write(yolo_annotation)
    
    def create_yaml_config(self, class_counts):
        """Tạo YAML configuration file"""
        print("\n📝 Creating YAML configuration...")
        
        # Create class names list (sorted by index)
        class_names = [""] * len(self.class_mapping)
        for class_folder, info in self.class_mapping.items():
            class_names[info["index"]] = info["name"]
        
        yaml_content = {
            'path': str(self.output_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': class_names
        }
        
        # Save YAML
        yaml_path = self.output_root / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Created {yaml_path}")
        
        # Save detailed statistics
        stats = {
            'total_images': sum(class_counts.values()),
            'num_classes': len(class_names),
            'class_distribution': dict(class_counts),
            'train_val_ratio': f"{self.train_ratio:.0%}/{1-self.train_ratio:.0%}"
        }
        
        stats_path = self.output_root / "dataset_stats.yaml"
        with open(stats_path, 'w', encoding='utf-8') as f:
            yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Created {stats_path}")
        
        return yaml_content
    
    def convert(self):
        """Main conversion process"""
        print("🚧 ROAD ISSUES DATASET CONVERTER")
        print("=" * 50)
        print(f"📂 Input: {self.input_root}")
        print(f"📂 Output: {self.output_root}")
        
        # Clean output directory
        if self.output_root.exists():
            print(f"🗑️ Cleaning existing output directory...")
            shutil.rmtree(self.output_root)
        
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Discover classes
        discovered_classes = self.discover_classes()
        
        # Collect all images
        all_images, class_counts = self.collect_images()
        
        if not all_images:
            print("❌ No images found! Check input directory structure.")
            return
        
        # Convert to YOLO format
        self.split_and_convert(all_images)
        
        # Create configuration
        yaml_config = self.create_yaml_config(class_counts)
        
        # Print summary
        print(f"\n✅ CONVERSION COMPLETE!")
        print(f"📊 Total: {len(all_images)} images")
        print(f"🏷️ Classes: {yaml_config['nc']}")
        print(f"📁 Output: {self.output_root}")
        
        # Print class mapping
        print(f"\n📋 Class Mapping:")
        for i, class_name in enumerate(yaml_config['names']):
            count = class_counts.get(class_name, 0)
            print(f"   {i}: {class_name} ({count} images)")

if __name__ == "__main__":
    converter = RoadIssuesConverter()
    converter.convert()