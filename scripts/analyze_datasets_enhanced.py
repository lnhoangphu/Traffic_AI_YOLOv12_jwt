"""
Enhanced Dataset Analysis Tool - Phân tích chi tiết và chính xác hơn
Đọc được classes thực sự từ YOLO labels và classification folders

Usage: python scripts/analyze_datasets_enhanced.py
"""

import os
import json
import yaml
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
from tabulate import tabulate
import re

class EnhancedDatasetAnalyzer:
    def __init__(self, datasets_root="datasets_src"):
        self.datasets_root = Path(datasets_root)
        self.results = []
        
    def count_files_by_extension(self, directory, extensions):
        """Đếm file theo extension trong thư mục"""
        if not directory.exists():
            return 0
        
        count = 0
        for ext in extensions:
            count += len(list(directory.glob(f"*.{ext}")))
            count += len(list(directory.glob(f"*.{ext.upper()}")))
        return count
    
    def read_classes_from_yolo_config(self, dataset_path):
        """Đọc classes từ YOLO config files"""
        config_files = [
            dataset_path / 'data.yaml',
            dataset_path / 'dataset.yaml', 
            dataset_path / 'config.yaml',
            dataset_path / 'custom_data.yaml',
            dataset_path / 'intersection.yaml',
            dataset_path / 'classes.txt'
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    print(f"      🔍 Reading config: {config_file.name}")
                    if config_file.suffix == '.txt':
                        with open(config_file, 'r', encoding='utf-8') as f:
                            classes = [line.strip() for line in f if line.strip()]
                        if classes:
                            print(f"         Found {len(classes)} classes: {classes}")
                            return classes
                    else:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f)
                            if data and 'names' in data:
                                if isinstance(data['names'], dict):
                                    class_list = [data['names'][i] for i in sorted(data['names'].keys())]
                                    print(f"         Found {len(class_list)} classes: {class_list}")
                                    return class_list
                                elif isinstance(data['names'], list):
                                    print(f"         Found {len(data['names'])} classes: {data['names']}")
                                    return data['names']
                except Exception as e:
                    print(f"      ⚠️ Error reading {config_file}: {e}")
        
        print(f"      ⚠️ No class config found in {dataset_path}")
        return []
    
    def analyze_yolo_labels_detailed(self, labels_dir):
        """Phân tích chi tiết YOLO labels"""
        if not labels_dir.exists():
            return set(), 0, {}
        
        class_counts = Counter()
        total_objects = 0
        files_with_labels = 0
        
        for label_file in labels_dir.glob("*.txt"):
            file_has_labels = False
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:  # YOLO format: class x y w h
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                                total_objects += 1
                                file_has_labels = True
                if file_has_labels:
                    files_with_labels += 1
            except Exception:
                continue
        
        print(f"      📊 Found {total_objects} objects in {files_with_labels} files")
        if class_counts:
            print(f"         Classes found: {sorted(class_counts.keys())}")
            for class_id in sorted(class_counts.keys()):
                print(f"           Class {class_id}: {class_counts[class_id]} objects")
        
        return set(class_counts.keys()), total_objects, dict(class_counts)
    
    def analyze_classification_folders(self, data_path):
        """Phân tích folder-based classification để tìm classes thực sự"""
        img_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
        
        # Tìm các thư mục con chứa ảnh
        class_folders = []
        class_counts = {}
        
        def explore_directory(path, level=0):
            if level > 3:  # Giới hạn depth để tránh infinite loop
                return
            
            for item in path.iterdir():
                if item.is_dir():
                    # Đếm ảnh trong thư mục này
                    img_count = sum(len(list(item.glob(f"*.{ext}"))) for ext in img_extensions)
                    img_count += sum(len(list(item.glob(f"*.{ext.upper()}"))) for ext in img_extensions)
                    
                    if img_count > 0:
                        # Thư mục này chứa ảnh - có thể là class
                        parent_name = item.parent.name.lower()
                        if parent_name not in ['train', 'val', 'test', 'training', 'validation', 'testing', 'images']:
                            class_folders.append(item.name)
                            class_counts[item.name] = img_count
                    
                    # Tiếp tục explore nếu không phải train/val/test
                    if item.name.lower() not in ['train', 'val', 'test', 'training', 'validation', 'testing']:
                        explore_directory(item, level + 1)
        
        explore_directory(data_path)
        
        return class_folders, class_counts
    
    def analyze_vn_traffic_sign(self, dataset_path):
        """Phân tích đặc biệt cho VN Traffic Sign dataset"""
        result = {
            'train_images': 0, 'val_images': 0, 'test_images': 0,
            'train_labels': 0, 'val_labels': 0, 'test_labels': 0,
            'classes': [], 'total_annotations': 0, 'class_distribution': {}
        }
        
        # Đọc classes từ config
        classes = self.read_classes_from_yolo_config(dataset_path)
        if classes:
            result['classes'] = classes
        
        # Phân tích từng split
        img_extensions = ['jpg', 'jpeg', 'png', 'bmp']
        
        for split in ['train', 'val', 'test']:
            img_dir = dataset_path / 'images' / split
            label_dir = dataset_path / 'labels' / split
            
            if img_dir.exists():
                result[f'{split}_images'] = self.count_files_by_extension(img_dir, img_extensions)
            
            if label_dir.exists():
                result[f'{split}_labels'] = self.count_files_by_extension(label_dir, ['txt'])
                
                # Phân tích classes trong labels
                classes_found, objects, class_counts = self.analyze_yolo_labels_detailed(label_dir)
                result['total_annotations'] += objects
                for class_id, count in class_counts.items():
                    result['class_distribution'][class_id] = result['class_distribution'].get(class_id, 0) + count
        
        return result
    
    def analyze_intersection_flow(self, dataset_path):
        """Phân tích đặc biệt cho Intersection Flow dataset"""
        result = {
            'train_images': 0, 'val_images': 0, 'test_images': 0,
            'train_labels': 0, 'val_labels': 0, 'test_labels': 0,
            'classes': [], 'total_annotations': 0, 'class_distribution': {}
        }
        
        # Đọc classes từ config
        classes = self.read_classes_from_yolo_config(dataset_path)
        if classes:
            result['classes'] = classes
        
        # Phân tích YOLO structure
        img_extensions = ['jpg', 'jpeg', 'png', 'bmp']
        
        for split in ['train', 'val', 'test']:
            img_dir = dataset_path / 'images' / split
            label_dir = dataset_path / 'labels' / split
            
            if img_dir.exists():
                result[f'{split}_images'] = self.count_files_by_extension(img_dir, img_extensions)
            
            if label_dir.exists():
                result[f'{split}_labels'] = self.count_files_by_extension(label_dir, ['txt'])
                
                # Phân tích classes
                classes_found, objects, class_counts = self.analyze_yolo_labels_detailed(label_dir)
                result['total_annotations'] += objects
                for class_id, count in class_counts.items():
                    result['class_distribution'][class_id] = result['class_distribution'].get(class_id, 0) + count
        
        # Nếu không có classes từ config, tạo từ labels
        if not result['classes'] and result['class_distribution']:
            max_class_id = max(result['class_distribution'].keys())
            result['classes'] = [f"class_{i}" for i in range(max_class_id + 1)]
        
        return result
    
    def analyze_road_issues(self, dataset_path):
        """Phân tích đặc biệt cho Road Issues dataset"""
        result = {
            'train_images': 0, 'val_images': 0, 'test_images': 0,
            'train_labels': 0, 'val_labels': 0, 'test_labels': 0,
            'classes': [], 'total_annotations': 0, 'class_distribution': {}
        }
        
        # Road Issues là classification dataset
        class_folders, class_counts = self.analyze_classification_folders(dataset_path)
        
        result['classes'] = class_folders
        result['train_images'] = sum(class_counts.values())
        result['class_distribution'] = class_counts
        
        return result
    
    def analyze_object_detection_35(self, dataset_path):
        """Phân tích đặc biệt cho Object Detection 35 dataset"""
        result = {
            'train_images': 0, 'val_images': 0, 'test_images': 0,
            'train_labels': 0, 'val_labels': 0, 'test_labels': 0,
            'classes': [], 'total_annotations': 0, 'class_distribution': {}
        }
        
        # Object Detection 35 có structure batch-based với train/val/test splits
        img_extensions = ['jpg', 'jpeg', 'png', 'bmp']
        
        # Duyệt các batch
        for batch_dir in dataset_path.glob("Batch *"):
            if batch_dir.is_dir():
                print(f"   🔍 Processing {batch_dir.name}...")
                
                # Duyệt train/val/test trong mỗi batch
                for split in ['train', 'val', 'test']:
                    img_dir = batch_dir / 'images' / split
                    label_dir = batch_dir / 'labels' / split
                    
                    if img_dir.exists():
                        batch_split_images = self.count_files_by_extension(img_dir, img_extensions)
                        result[f'{split}_images'] += batch_split_images
                        print(f"       {split}: {batch_split_images} images")
                    
                    if label_dir.exists():
                        batch_split_labels = self.count_files_by_extension(label_dir, ['txt'])
                        result[f'{split}_labels'] += batch_split_labels
                        
                        # Phân tích classes trong batch split này
                        classes_found, objects, class_counts = self.analyze_yolo_labels_detailed(label_dir)
                        result['total_annotations'] += objects
                        for class_id, count in class_counts.items():
                            result['class_distribution'][class_id] = result['class_distribution'].get(class_id, 0) + count
        
        # Tạo class names dựa trên classes tìm được
        if result['class_distribution']:
            max_class_id = max(result['class_distribution'].keys())
            # Object Detection 35 là cutlery dataset với 35 classes
            cutlery_classes = [
                'bowl', 'plate', 'fork', 'knife', 'spoon', 'cup', 'glass', 'chopsticks',
                'saucer', 'tray', 'napkin', 'placemat', 'tablecloth', 'coaster', 'pitcher',
                'teapot', 'sugar_bowl', 'cream_pitcher', 'salt_shaker', 'pepper_shaker',
                'butter_dish', 'serving_spoon', 'ladle', 'spatula', 'tongs', 'whisk',
                'cutting_board', 'rolling_pin', 'can_opener', 'bottle_opener', 'corkscrew',
                'grater', 'strainer', 'colander', 'mixer'
            ]
            
            # Sử dụng tên cutlery nếu có đủ, không thì dùng generic names
            if max_class_id < len(cutlery_classes):
                result['classes'] = cutlery_classes[:max_class_id + 1]
            else:
                result['classes'] = [f"cutlery_item_{i}" for i in range(max_class_id + 1)]
        
        return result
    
    def analyze_dataset(self, dataset_name, dataset_path):
        """Phân tích dataset với logic đặc biệt cho từng loại"""
        print(f"\n🔍 Analyzing {dataset_name}...")
        
        if dataset_name == 'vn_traffic_sign':
            actual_path = dataset_path / 'dataset'
            if actual_path.exists():
                result = self.analyze_vn_traffic_sign(actual_path)
            else:
                result = self.analyze_vn_traffic_sign(dataset_path)
        
        elif dataset_name == 'intersection_flow_5k':
            actual_path = dataset_path / 'Intersection-Flow-5K'
            if actual_path.exists():
                result = self.analyze_intersection_flow(actual_path)
            else:
                result = self.analyze_intersection_flow(dataset_path)
        
        elif dataset_name == 'road_issues':
            actual_path = dataset_path / 'data'
            if actual_path.exists():
                result = self.analyze_road_issues(actual_path)
            else:
                result = self.analyze_road_issues(dataset_path)
        
        elif dataset_name == 'object_detection_35':
            actual_path = dataset_path / 'final batches'
            if actual_path.exists():
                result = self.analyze_object_detection_35(actual_path)
            else:
                result = self.analyze_object_detection_35(dataset_path)
        
        else:
            # Generic analysis
            result = {
                'train_images': 0, 'val_images': 0, 'test_images': 0,
                'train_labels': 0, 'val_labels': 0, 'test_labels': 0,
                'classes': [], 'total_annotations': 0, 'class_distribution': {}
            }
        
        # Add metadata
        result['dataset'] = dataset_name
        result['path'] = str(dataset_path)
        
        # Add quality comments
        self.add_enhanced_comments(result)
        
        return result
    
    def add_enhanced_comments(self, result):
        """Thêm nhận xét chi tiết về dataset"""
        comments = []
        
        total_images = result['train_images'] + result['val_images'] + result['test_images']
        total_labels = result['train_labels'] + result['val_labels'] + result['test_labels']
        
        # Dataset size assessment
        if total_images == 0:
            comments.append("❌ Không có ảnh")
        elif total_images < 100:
            comments.append("⚠️ Dataset nhỏ (<100 ảnh)")
        elif total_images < 1000:
            comments.append("🟡 Dataset trung bình (<1k ảnh)")
        elif total_images < 10000:
            comments.append("🟢 Dataset khá lớn (<10k ảnh)")
        else:
            comments.append("✅ Dataset lớn (>10k ảnh)")
        
        # Train/Val/Test split assessment
        if result['val_images'] > 0 and result['test_images'] > 0:
            comments.append("✅ Có đầy đủ train/val/test")
        elif result['val_images'] > 0:
            comments.append("✅ Có train/val split")
        elif result['test_images'] > 0:
            comments.append("🟡 Có train/test split")
        else:
            comments.append("⚠️ Chỉ có train data")
        
        # Label coverage assessment
        if total_labels > 0:
            label_coverage = total_labels / max(total_images, 1)
            if label_coverage >= 0.95:
                comments.append("✅ Labels đầy đủ")
            elif label_coverage >= 0.8:
                comments.append("🟢 Labels khá đầy đủ")
            elif label_coverage >= 0.5:
                comments.append("🟡 Thiếu một số labels")
            else:
                comments.append("⚠️ Thiếu nhiều labels")
        
        # Class assessment
        num_classes = len(result['classes'])
        if num_classes == 0:
            comments.append("❌ Không xác định được classes")
        elif num_classes == 1:
            comments.append("ℹ️ Single-class dataset")
        elif num_classes <= 10:
            comments.append(f"✅ {num_classes} classes (balanced)")
        elif num_classes <= 50:
            comments.append(f"🟡 {num_classes} classes (multi-class)")
        else:
            comments.append(f"⚠️ {num_classes} classes (rất nhiều)")
        
        # Annotation density
        if result['total_annotations'] > 0 and total_images > 0:
            annotations_per_image = result['total_annotations'] / total_images
            if annotations_per_image < 0.1:
                comments.append("⚠️ Rất ít annotations/ảnh")
            elif annotations_per_image < 1:
                comments.append("🟡 Ít annotations/ảnh")
            elif annotations_per_image <= 5:
                comments.append("✅ Annotations/ảnh hợp lý")
            else:
                comments.append("ℹ️ Nhiều objects/ảnh")
        
        # Class distribution assessment
        if result['class_distribution']:
            class_counts = list(result['class_distribution'].values())
            if len(class_counts) > 1:
                max_count = max(class_counts)
                min_count = min(class_counts)
                imbalance_ratio = max_count / max(min_count, 1)
                
                if imbalance_ratio > 10:
                    comments.append("⚠️ Classes mất cân bằng nghiêm trọng")
                elif imbalance_ratio > 3:
                    comments.append("🟡 Classes hơi mất cân bằng")
                else:
                    comments.append("✅ Classes cân bằng")
        
        result['comments'] = comments
    
    def analyze_all_datasets(self):
        """Phân tích tất cả datasets"""
        print("🚀 ENHANCED DATASET ANALYSIS TOOL")
        print("=" * 70)
        
        if not self.datasets_root.exists():
            print(f"❌ Thư mục {self.datasets_root} không tồn tại!")
            return
        
        # Find dataset directories
        dataset_dirs = [d for d in self.datasets_root.iterdir() if d.is_dir()]
        
        if not dataset_dirs:
            print(f"❌ Không tìm thấy dataset nào trong {self.datasets_root}")
            return
        
        print(f"📁 Tìm thấy {len(dataset_dirs)} datasets:")
        for d in dataset_dirs:
            print(f"   - {d.name}")
        
        # Analyze each dataset
        for dataset_dir in dataset_dirs:
            result = self.analyze_dataset(dataset_dir.name, dataset_dir)
            self.results.append(result)
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self):
        """Tạo báo cáo toàn diện"""
        print("\n📊 COMPREHENSIVE DATASET REPORT")
        print("=" * 80)
        
        # Summary table
        table_data = []
        for result in self.results:
            total_images = result['train_images'] + result['val_images'] + result['test_images']
            total_labels = result['train_labels'] + result['val_labels'] + result['test_labels']
            
            # Format image breakdown
            img_detail = f"{total_images}"
            if result['val_images'] > 0 or result['test_images'] > 0:
                img_detail += f" (T:{result['train_images']}/V:{result['val_images']}/Te:{result['test_images']})"
            
            # Format label info
            label_detail = f"{total_labels}"
            if total_labels > 0:
                coverage = (total_labels / max(total_images, 1)) * 100
                label_detail += f" ({coverage:.0f}%)"
            
            # Classes summary
            num_classes = len(result['classes'])
            if num_classes <= 5 and result['classes']:
                class_summary = f"{num_classes}: {', '.join(result['classes'][:3])}"
                if num_classes > 3:
                    class_summary += "..."
            else:
                class_summary = f"{num_classes} classes"
            
            # Top comments
            top_comments = "; ".join(result['comments'][:2])
            
            table_data.append([
                result['dataset'],
                img_detail,
                label_detail,
                class_summary,
                result['total_annotations'],
                top_comments
            ])
        
        headers = ["Dataset", "Images (T/V/Te)", "Labels (%)", "Classes", "Annotations", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[15, 18, 15, 25, 12, 35]))
        
        # Detailed breakdown for each dataset
        self.print_detailed_analysis()
    
    def print_detailed_analysis(self):
        """In phân tích chi tiết cho từng dataset"""
        print("\n📋 DETAILED ANALYSIS")
        print("=" * 80)
        
        for result in self.results:
            print(f"\n🎯 {result['dataset'].upper()}")
            print(f"   📁 Path: {result['path']}")
            
            # Image statistics
            total_images = result['train_images'] + result['val_images'] + result['test_images']
            print(f"   🖼️ Images: {total_images} total")
            if result['train_images'] > 0:
                print(f"       └── Train: {result['train_images']}")
            if result['val_images'] > 0:
                print(f"       └── Val: {result['val_images']}")
            if result['test_images'] > 0:
                print(f"       └── Test: {result['test_images']}")
            
            # Label statistics
            total_labels = result['train_labels'] + result['val_labels'] + result['test_labels']
            if total_labels > 0:
                print(f"   🏷️ Labels: {total_labels} total ({(total_labels/max(total_images,1)*100):.1f}% coverage)")
                print(f"   📊 Annotations: {result['total_annotations']} objects ({result['total_annotations']/max(total_images,1):.1f} obj/img)")
            
            # Classes
            print(f"   🎯 Classes ({len(result['classes'])}):")
            if result['classes']:
                for i, class_name in enumerate(result['classes'][:10]):
                    count = result['class_distribution'].get(i, 0) if isinstance(list(result['class_distribution'].keys())[0] if result['class_distribution'] else 0, int) else result['class_distribution'].get(class_name, 0)
                    print(f"       {i}: {class_name} ({count} samples)")
                if len(result['classes']) > 10:
                    print(f"       ... và {len(result['classes']) - 10} classes khác")
            else:
                print("       Không xác định được classes")
            
            # Comments
            print(f"   💭 Assessment:")
            for comment in result['comments']:
                print(f"       {comment}")

def main():
    analyzer = EnhancedDatasetAnalyzer()
    analyzer.analyze_all_datasets()
    
    print("\n🎉 Enhanced analysis complete!")
    print("\n📚 TRAINING RECOMMENDATIONS:")
    print("=" * 50)
    
    for result in analyzer.results:
        print(f"\n🎯 {result['dataset']}:")
        if result['dataset'] == 'vn_traffic_sign':
            print("   ✅ Ready for YOLO training - Good structure")
            print("   📝 Use: python scripts/train_vn_traffic_sign.py")
        elif result['dataset'] == 'intersection_flow_5k':
            print("   ✅ Ready for YOLO training - Multi-class traffic")
            print("   📝 Use: python scripts/train_intersection_flow.py")
        elif result['dataset'] == 'road_issues':
            print("   🔄 Needs conversion to YOLO format")
            print("   📝 Use: python scripts/train_road_issues.py")
        elif result['dataset'] == 'object_detection_35':
            print("   ⚠️ Not traffic-related (cutlery dataset)")
            print("   📝 Skip or use for testing only")

if __name__ == "__main__":
    main()