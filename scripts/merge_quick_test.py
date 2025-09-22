"""
Quick Test Merger - Test with smaller samples from each dataset
Kiểm tra với samples nhỏ để verify taxonomy hoạt động đúng

Usage: python scripts/merge_quick_test.py
"""

import os
import sys
import shutil
import random
import yaml
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

class QuickTestMerger:
    def __init__(self):
        self.output_root = Path("datasets/traffic_ai_test_11class")
        
        # Load taxonomy config
        self.load_taxonomy_config()
        
        # Dataset paths (giữ nguyên format gốc)
        self.dataset_paths = {
            'object_detection_35': Path("datasets_src/object_detection_35_organized"),
            'intersection_flow_5k': Path("datasets_src/intersection_flow_5k/Intersection-Flow-5K"),
            'vn_traffic_sign': Path("datasets_src/vn_traffic_sign/dataset"),
            'road_issues': Path("datasets_src/road_issues_yolo")
        }
        
        # Test with small samples
        self.max_samples_per_dataset = 100
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'input_annotations': 0,
            'skipped_annotations': 0,
            'class_distribution': Counter(),
            'dataset_contributions': defaultdict(lambda: defaultdict(int)),
            'split_distribution': defaultdict(int)
        }

    def load_taxonomy_config(self):
        """Load taxonomy configuration"""
        config_path = Path("config/taxonomy_complete_11class.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print(f"✅ Loaded taxonomy config: {len(self.config['classes'])} classes")
        for i, cls in enumerate(self.config['classes']):
            print(f"   {i}: {cls}")

    def convert_annotation_line(self, line, dataset_name):
        """Convert annotation line using dataset-specific mapping"""
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        
        # Handle float class IDs by converting to int
        try:
            old_class_id = int(float(parts[0]))  # Convert float to int
        except ValueError:
            self.stats['skipped_annotations'] += 1
            return None
            
        bbox_info = parts[1:5]
        
        self.stats['input_annotations'] += 1
        
        # Get mapping for this dataset
        if dataset_name not in self.config['mapping']:
            self.stats['skipped_annotations'] += 1
            return None
        
        dataset_mapping = self.config['mapping'][dataset_name]
        
        # Handle special case for vn_traffic_sign (all classes -> traffic_sign)
        if dataset_name == 'vn_traffic_sign':
            new_class_id = 10  # Traffic Sign in user's taxonomy
        elif old_class_id in dataset_mapping:
            new_class_id = dataset_mapping[old_class_id]
        else:
            # Skip unmapped classes
            self.stats['skipped_annotations'] += 1
            return None
        
        self.stats['class_distribution'][new_class_id] += 1
        self.stats['dataset_contributions'][dataset_name][new_class_id] += 1
        return f"{new_class_id} {' '.join(bbox_info)}"

    def process_dataset_file(self, img_path, label_path, dataset_name, target_split, file_index):
        """Process single image-label pair"""
        try:
            # Read and convert annotations
            converted_annotations = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    converted_line = self.convert_annotation_line(line, dataset_name)
                    if converted_line:
                        converted_annotations.append(converted_line)
            
            # Skip if no valid annotations
            if not converted_annotations:
                return False
            
            # Create output filenames
            dataset_prefix = {
                'object_detection_35': 'od35',
                'intersection_flow_5k': 'inter',
                'vn_traffic_sign': 'sign',
                'road_issues': 'road'
            }[dataset_name]
            
            new_name = f"{dataset_prefix}_{file_index:06d}"
            
            # Copy image
            output_img_path = self.output_root / 'images' / target_split / f"{new_name}.jpg"
            shutil.copy2(img_path, output_img_path)
            
            # Write converted annotations
            output_label_path = self.output_root / 'labels' / target_split / f"{new_name}.txt"
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(converted_annotations))
            
            self.stats['total_images'] += 1
            self.stats['total_annotations'] += len(converted_annotations)
            self.stats['split_distribution'][target_split] += 1
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            return False

    def find_dataset_files(self, dataset_name, dataset_path):
        """Find all image-label pairs in dataset"""
        files = []
        
        if dataset_name == 'object_detection_35':
            # Object Detection 35 structure
            for split in ['train', 'val']:
                img_dir = dataset_path / 'images' / split
                label_dir = dataset_path / 'labels' / split
                
                if img_dir.exists() and label_dir.exists():
                    for img_file in img_dir.glob('*.jpg'):
                        label_file = label_dir / f"{img_file.stem}.txt"
                        if label_file.exists():
                            files.append({
                                'img_path': img_file,
                                'label_path': label_file,
                                'original_split': split
                            })
        
        elif dataset_name == 'intersection_flow_5k':
            # Try to find structure
            for root, dirs, file_list in os.walk(dataset_path):
                root_path = Path(root)
                for file in file_list:
                    if file.endswith('.jpg'):
                        img_path = root_path / file
                        label_path = root_path / f"{Path(file).stem}.txt"
                        if label_path.exists():
                            files.append({
                                'img_path': img_path,
                                'label_path': label_path,
                                'original_split': 'train'  # Default split
                            })
        
        elif dataset_name == 'vn_traffic_sign':
            # VN Traffic Sign structure
            for split in ['train', 'val', 'test']:
                img_dir = dataset_path / 'images' / split
                label_dir = dataset_path / 'labels' / split
                
                if img_dir.exists() and label_dir.exists():
                    for img_file in img_dir.glob('*.jpg'):
                        label_file = label_dir / f"{img_file.stem}.txt"
                        if label_file.exists():
                            files.append({
                                'img_path': img_file,
                                'label_path': label_file,
                                'original_split': split
                            })
        
        elif dataset_name == 'road_issues':
            # Road Issues structure  
            for split in ['train', 'val', 'test']:
                img_dir = dataset_path / 'images' / split
                label_dir = dataset_path / 'labels' / split
                
                if img_dir.exists() and label_dir.exists():
                    for img_file in img_dir.glob('*.jpg'):
                        label_file = label_dir / f"{img_file.stem}.txt"
                        if label_file.exists():
                            files.append({
                                'img_path': img_file,
                                'label_path': label_file,
                                'original_split': split
                            })
        
        return files

    def process_dataset(self, dataset_name, max_samples=None):
        """Process one dataset"""
        dataset_path = self.dataset_paths[dataset_name]
        
        if not dataset_path.exists():
            print(f"⚠️ Dataset path not found: {dataset_path}")
            return
        
        print(f"\n📂 Processing {dataset_name} from {dataset_path}")
        
        # Find all files
        files = self.find_dataset_files(dataset_name, dataset_path)
        print(f"   Found {len(files)} image-label pairs")
        
        if not files:
            print(f"   No valid files found in {dataset_name}")
            return
        
        # Limit for testing
        if max_samples and len(files) > max_samples:
            files = random.sample(files, max_samples)
            print(f"   Testing with {len(files)} samples")
        
        # Process files
        success_count = 0
        file_index = 0
        
        for file_info in tqdm(files, desc=f"Processing {dataset_name}"):
            # Determine target split (simplified for test)
            target_split = 'train' if random.random() < 0.8 else 'val'
            
            success = self.process_dataset_file(
                file_info['img_path'],
                file_info['label_path'],
                dataset_name,
                target_split,
                file_index
            )
            
            if success:
                success_count += 1
            file_index += 1
        
        print(f"   ✅ Successfully processed: {success_count}/{len(files)} files")

    def merge(self):
        """Main merge function"""
        print("🚀 STARTING QUICK TEST MERGE")
        print("=" * 60)
        
        # Create output directory
        if self.output_root.exists():
            shutil.rmtree(self.output_root)
        
        # Create directories
        for split in ['train', 'val']:
            (self.output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Process each dataset
        for dataset_name in ['object_detection_35', 'vn_traffic_sign', 'road_issues']:
            self.process_dataset(dataset_name, self.max_samples_per_dataset)
        
        # Save statistics
        self.save_final_report()
        
        print("\n🎯 QUICK TEST MERGE COMPLETED!")

    def save_final_report(self):
        """Save final statistics and create data.yaml"""
        
        # Create data.yaml for YOLO training
        data_yaml = {
            'path': str(self.output_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.config['classes']),
            'names': self.config['classes']
        }
        
        with open(self.output_root / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        
        # Statistics report
        report = f"""
🎯 QUICK TEST MERGE REPORT
{'=' * 50}

📊 DATASET STATISTICS:
   Total Images: {self.stats['total_images']}
   Total Annotations: {self.stats['total_annotations']}
   Input Annotations: {self.stats['input_annotations']}
   Skipped Annotations: {self.stats['skipped_annotations']}
   
📈 CLASS DISTRIBUTION:
"""
        
        for class_id in sorted(self.stats['class_distribution'].keys()):
            class_name = self.config['classes'][class_id]
            count = self.stats['class_distribution'][class_id]
            report += f"   {class_id}: {class_name:<15} = {count:>5} annotations\n"
        
        report += f"\n📂 SPLIT DISTRIBUTION:\n"
        for split, count in self.stats['split_distribution'].items():
            report += f"   {split}: {count} images\n"
        
        report += f"\n🔗 DATASET CONTRIBUTIONS:\n"
        for dataset, classes in self.stats['dataset_contributions'].items():
            report += f"   {dataset}:\n"
            for class_id, count in sorted(classes.items()):
                class_name = self.config['classes'][class_id]
                report += f"      {class_id}: {class_name} = {count} annotations\n"
        
        print(report)
        
        # Save report to file
        with open(self.output_root / 'merge_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

if __name__ == "__main__":
    merger = QuickTestMerger()
    merger.merge()