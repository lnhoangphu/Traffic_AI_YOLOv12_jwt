"""
Dataset Verification Script - Kiểm tra các dataset đã chuyển đổi
Verify converted datasets: Road Issues và Object Detection 35

Usage: python scripts/verify_converted_datasets.py
"""

import os
import yaml
from pathlib import Path
from collections import Counter
import random

class DatasetVerifier:
    def __init__(self):
        self.datasets = {
            'road_issues_yolo': Path('datasets_src/road_issues_yolo'),
            'object_detection_35_yolo': Path('datasets_src/object_detection_35_yolo')
        }
    
    def verify_directory_structure(self, dataset_path):
        """Kiểm tra cấu trúc thư mục"""
        print(f"📁 Checking directory structure for {dataset_path.name}...")
        
        required_dirs = [
            'images/train',
            'images/val', 
            'labels/train',
            'labels/val'
        ]
        
        required_files = [
            'data.yaml',
            'dataset_stats.yaml'
        ]
        
        missing_dirs = []
        missing_files = []
        
        for dir_path in required_dirs:
            full_path = dataset_path / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
            else:
                print(f"   ✅ {dir_path}")
        
        for file_path in required_files:
            full_path = dataset_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                print(f"   ✅ {file_path}")
        
        if missing_dirs or missing_files:
            print(f"   ❌ Missing directories: {missing_dirs}")
            print(f"   ❌ Missing files: {missing_files}")
            return False
        
        return True
    
    def count_files(self, dataset_path):
        """Đếm số lượng file trong dataset"""
        print(f"📊 Counting files in {dataset_path.name}...")
        
        counts = {}
        
        for split in ['train', 'val']:
            img_dir = dataset_path / 'images' / split
            label_dir = dataset_path / 'labels' / split
            
            img_count = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
            label_count = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0
            
            counts[split] = {
                'images': img_count,
                'labels': label_count,
                'match': img_count == label_count
            }
            
            status = "✅" if img_count == label_count else "❌"
            print(f"   {status} {split}: {img_count} images, {label_count} labels")
        
        return counts
    
    def verify_yaml_config(self, dataset_path):
        """Kiểm tra YAML configuration"""
        print(f"📝 Verifying YAML config for {dataset_path.name}...")
        
        yaml_path = dataset_path / 'data.yaml'
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            required_keys = ['path', 'train', 'val', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                print(f"   ❌ Missing keys: {missing_keys}")
                return False
            
            print(f"   ✅ Path: {config['path']}")
            print(f"   ✅ Train: {config['train']}")
            print(f"   ✅ Val: {config['val']}")
            print(f"   ✅ Classes: {config['nc']}")
            print(f"   ✅ Class names: {len(config['names'])} names")
            
            # Verify class count matches
            if len(config['names']) != config['nc']:
                print(f"   ❌ Class count mismatch: nc={config['nc']}, names={len(config['names'])}")
                return False
            
            return config
            
        except Exception as e:
            print(f"   ❌ Error reading YAML: {e}")
            return False
    
    def sample_annotations(self, dataset_path, num_samples=3):
        """Lấy mẫu annotations để kiểm tra format"""
        print(f"🔍 Sampling annotations from {dataset_path.name}...")
        
        label_dir = dataset_path / 'labels' / 'train'
        label_files = list(label_dir.glob('*.txt'))
        
        if not label_files:
            print("   ❌ No label files found!")
            return False
        
        sample_files = random.sample(label_files, min(num_samples, len(label_files)))
        
        for i, label_file in enumerate(sample_files):
            print(f"   📄 Sample {i+1}: {label_file.name}")
            
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for j, line in enumerate(lines[:3]):  # Show first 3 annotations
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x, y, w, h = parts[:5]
                        print(f"      Line {j+1}: class={class_id}, center=({x},{y}), size=({w},{h})")
                    else:
                        print(f"      Line {j+1}: Invalid format - {line.strip()}")
                
                if len(lines) > 3:
                    print(f"      ... and {len(lines)-3} more annotations")
                    
            except Exception as e:
                print(f"   ❌ Error reading {label_file.name}: {e}")
        
        return True
    
    def analyze_class_distribution(self, dataset_path, config):
        """Phân tích phân phối class"""
        print(f"📈 Analyzing class distribution for {dataset_path.name}...")
        
        class_counts = Counter()
        total_annotations = 0
        
        for split in ['train', 'val']:
            label_dir = dataset_path / 'labels' / split
            
            if not label_dir.exists():
                continue
            
            for label_file in label_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))
                                if 0 <= class_id < len(config['names']):
                                    class_name = config['names'][class_id]
                                    class_counts[class_name] += 1
                                    total_annotations += 1
                except Exception:
                    continue
        
        print(f"   📊 Total annotations: {total_annotations}")
        print(f"   🏷️ Top 10 classes:")
        
        for class_name, count in class_counts.most_common(10):
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"      {class_name}: {count} ({percentage:.1f}%)")
        
        return class_counts, total_annotations
    
    def verify_dataset(self, dataset_name, dataset_path):
        """Verify single dataset"""
        print(f"\n🚀 VERIFYING {dataset_name.upper()}")
        print("=" * 60)
        
        if not dataset_path.exists():
            print(f"❌ Dataset path does not exist: {dataset_path}")
            return False
        
        # Check directory structure
        if not self.verify_directory_structure(dataset_path):
            return False
        
        # Count files
        file_counts = self.count_files(dataset_path)
        
        # Verify YAML config
        config = self.verify_yaml_config(dataset_path)
        if not config:
            return False
        
        # Sample annotations
        if not self.sample_annotations(dataset_path):
            return False
        
        # Analyze class distribution
        class_counts, total_annotations = self.analyze_class_distribution(dataset_path, config)
        
        # Summary
        total_images = sum(file_counts[split]['images'] for split in ['train', 'val'])
        total_labels = sum(file_counts[split]['labels'] for split in ['train', 'val'])
        
        print(f"\n📋 SUMMARY for {dataset_name}:")
        print(f"   📁 Total images: {total_images}")
        print(f"   🏷️ Total labels: {total_labels}")
        print(f"   📝 Total annotations: {total_annotations}")
        print(f"   🎯 Classes: {config['nc']}")
        print(f"   ✅ Status: {'VALID' if total_images == total_labels else 'INVALID'}")
        
        return True
    
    def verify_all(self):
        """Verify all converted datasets"""
        print("🔍 DATASET VERIFICATION TOOL")
        print("=" * 60)
        
        all_valid = True
        
        for dataset_name, dataset_path in self.datasets.items():
            valid = self.verify_dataset(dataset_name, dataset_path)
            all_valid = all_valid and valid
        
        print(f"\n🎉 VERIFICATION COMPLETE!")
        print(f"📊 Status: {'ALL DATASETS VALID' if all_valid else 'SOME ISSUES FOUND'}")
        
        return all_valid

if __name__ == "__main__":
    verifier = DatasetVerifier()
    verifier.verify_all()