"""
Final 11-Class Dataset Merger - CORRECT APPROACH
Gộp 4 datasets với class formats gốc và convert sang 11-class taxonomy chỉ trong bước merge

Datasets (giữ nguyên format gốc):
1. Object Detection 35 (35 classes) - ✅ 15,893 images, 22,792 annotations  
2. Intersection Flow 5K (8 classes) - Traffic flow detection
3. VN Traffic Sign (29 classes) - Vietnamese traffic signs  
4. Road Issues (7 classes) - Infrastructure problems

Output: datasets/traffic_ai_final_11class/ (11 classes)

Usage: python scripts/merge_datasets_final_correct.py
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

class FinalCorrectMerger:
    def __init__(self):
        self.output_root = Path("datasets/traffic_ai_final_11class")
        
        # Load taxonomy config
        self.load_taxonomy_config()
        
        # Dataset paths (giữ nguyên format gốc)
        self.dataset_paths = {
            'object_detection_35': Path("datasets_src/object_detection_35_organized"),
            'intersection_flow_5k': Path("datasets_src/intersection_flow_5k/Intersection-Flow-5K"),
            'vn_traffic_sign': Path("datasets_src/vn_traffic_sign/dataset"),
            'road_issues': Path("datasets_src/road_issues_yolo")
        }
        
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
        
        # Split ratios for final dataset
        self.split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    
    def load_taxonomy_config(self):
        """Load 11-class taxonomy configuration"""
        config_path = Path("config/taxonomy_complete_11class.yaml")
        
        if not config_path.exists():
            print(f"❌ Taxonomy config not found: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['classes']
        self.num_classes = len(self.class_names)
        
        print(f"✅ Loaded 11-class taxonomy: {self.num_classes} classes")
        for i, name in enumerate(self.class_names):
            print(f"   {i}: {name}")
    
    def setup_output_structure(self):
        """Create output directory structure"""
        print("\\n📁 Setting up output structure...")
        
        # Clean existing output
        if self.output_root.exists():
            shutil.rmtree(self.output_root)
        
        # Create directories
        for split in ['train', 'val', 'test']:
            (self.output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def convert_annotation_line(self, line, dataset_name):
        """Convert annotation line using dataset-specific mapping"""
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        
        old_class_id = int(parts[0])
        bbox_info = parts[1:5]
        
        self.stats['input_annotations'] += 1
        
        # Get mapping for this dataset
        if dataset_name not in self.config['mapping']:
            self.stats['skipped_annotations'] += 1
            return None
        
        dataset_mapping = self.config['mapping'][dataset_name]
        
        # Handle special case for vn_traffic_sign (all classes -> traffic_sign)
        if dataset_name == 'vn_traffic_sign':
            new_class_id = 8  # traffic_sign
        elif old_class_id in dataset_mapping:
            new_class_id = dataset_mapping[old_class_id]
        else:
            # Skip unmapped classes
            self.stats['skipped_annotations'] += 1
            return None
        
        return f"{new_class_id} {' '.join(bbox_info)}"
    
    def process_dataset_file(self, img_path, label_path, dataset_name, target_split, file_index):
        """Process single image-label pair"""
        try:
            # Read and convert annotations
            converted_annotations = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        converted_line = self.convert_annotation_line(line, dataset_name)
                        if converted_line:
                            converted_annotations.append(converted_line)
                            class_id = int(converted_line.split()[0])
                            self.stats['class_distribution'][class_id] += 1
                            self.stats['dataset_contributions'][dataset_name][class_id] += 1
            
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
                f.write('\\n'.join(converted_annotations))
            
            self.stats['total_images'] += 1
            self.stats['total_annotations'] += len(converted_annotations)
            self.stats['split_distribution'][target_split] += 1
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            return False
    
    def collect_dataset_files(self, dataset_name):
        """Collect all image-label pairs from a dataset"""
        dataset_path = self.dataset_paths[dataset_name]
        
        if not dataset_path.exists():
            print(f"⚠️ Dataset not found: {dataset_path}")
            return []
        
        print(f"📂 Collecting files from {dataset_name}...")
        
        all_files = []
        
        # Handle different dataset structures
        if dataset_name in ['object_detection_35', 'intersection_flow_5k', 'vn_traffic_sign']:
            # Have train/val/test splits
            for split in ['train', 'val', 'test']:
                img_dir = dataset_path / 'images' / split
                label_dir = dataset_path / 'labels' / split
                
                if img_dir.exists() and label_dir.exists():
                    for img_file in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
                        label_file = label_dir / f"{img_file.stem}.txt"
                        if label_file.exists():
                            all_files.append({
                                'img_path': img_file,
                                'label_path': label_file,
                                'dataset': dataset_name,
                                'original_split': split
                            })
        
        elif dataset_name == 'road_issues':
            # Has train/val splits only
            for split in ['train', 'val']:
                img_dir = dataset_path / 'images' / split
                label_dir = dataset_path / 'labels' / split
                
                if img_dir.exists() and label_dir.exists():
                    for img_file in img_dir.glob("*.jpg"):
                        label_file = label_dir / f"{img_file.stem}.txt"
                        if label_file.exists():
                            all_files.append({
                                'img_path': img_file,
                                'label_path': label_file,
                                'dataset': dataset_name,
                                'original_split': split
                            })
        
        print(f"   Found {len(all_files)} image-label pairs")
        return all_files
    
    def redistribute_files(self, all_files):
        """Redistribute files across train/val/test splits"""
        print(f"\\n🔄 Redistributing {len(all_files)} files...")
        
        # Shuffle for random distribution
        random.seed(42)
        random.shuffle(all_files)
        
        # Calculate split indices
        n_total = len(all_files)
        n_train = int(n_total * self.split_ratios['train'])
        n_val = int(n_total * self.split_ratios['val'])
        
        # Assign splits
        for i, file_info in enumerate(all_files):
            if i < n_train:
                file_info['target_split'] = 'train'
            elif i < n_train + n_val:
                file_info['target_split'] = 'val'
            else:
                file_info['target_split'] = 'test'
        
        # Print split distribution
        split_counts = Counter(f['target_split'] for f in all_files)
        for split in ['train', 'val', 'test']:
            count = split_counts[split]
            percentage = count / n_total * 100
            print(f"   {split:5s}: {count:5,d} files ({percentage:4.1f}%)")
        
        return all_files
    
    def process_all_datasets(self):
        """Process all datasets and merge"""
        print("\\n🔄 Processing all datasets...")
        
        all_files = []
        
        # Collect files from all datasets
        for dataset_name in self.dataset_paths.keys():
            dataset_files = self.collect_dataset_files(dataset_name)
            all_files.extend(dataset_files)
            
            # Print dataset info
            if dataset_files:
                dataset_path = self.dataset_paths[dataset_name]
                if (dataset_path / "data.yaml").exists():
                    with open(dataset_path / "data.yaml", 'r') as f:
                        dataset_config = yaml.safe_load(f)
                        print(f"      Original classes: {dataset_config.get('nc', 'unknown')}")
        
        if not all_files:
            print("❌ No files found in any dataset!")
            return
        
        print(f"\\n📊 Total files collected: {len(all_files)}")
        
        # Redistribute across train/val/test
        all_files = self.redistribute_files(all_files)
        
        # Process each file
        print(f"\\n📊 Processing and converting to 11-class taxonomy...")
        
        file_index = 0
        for file_info in tqdm(all_files, desc="Processing files"):
            success = self.process_dataset_file(
                file_info['img_path'],
                file_info['label_path'],
                file_info['dataset'],
                file_info['target_split'],
                file_index
            )
            if success:
                file_index += 1
    
    def create_dataset_config(self):
        """Create final dataset configuration"""
        print("\\n📝 Creating final dataset configuration...")
        
        # Create data.yaml with 11 classes
        yaml_content = {
            'path': str(self.output_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.num_classes,
            'names': self.class_names
        }
        
        yaml_path = self.output_root / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Created: {yaml_path}")
        
        # Create detailed statistics
        retention_rate = (self.stats['total_annotations'] / self.stats['input_annotations'] * 100) if self.stats['input_annotations'] > 0 else 0
        
        stats_content = {
            'merge_summary': {
                'total_images': self.stats['total_images'],
                'input_annotations': self.stats['input_annotations'],
                'output_annotations': self.stats['total_annotations'],
                'skipped_annotations': self.stats['skipped_annotations'],
                'retention_rate_percent': round(retention_rate, 1),
                'num_classes': self.num_classes,
                'datasets_merged': list(self.dataset_paths.keys())
            },
            'split_distribution': dict(self.stats['split_distribution']),
            'class_distribution': dict(self.stats['class_distribution']),
            'dataset_contributions': dict(self.stats['dataset_contributions']),
            'class_names': self.class_names,
            'taxonomy_mapping': self.config['mapping']
        }
        
        stats_path = self.output_root / "merge_statistics.yaml"
        with open(stats_path, 'w', encoding='utf-8') as f:
            yaml.dump(stats_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Created: {stats_path}")
    
    def print_final_summary(self):
        """Print comprehensive merge summary"""
        print("\\n🎉 FINAL MERGE SUMMARY")
        print("=" * 60)
        
        retention_rate = (self.stats['total_annotations'] / self.stats['input_annotations'] * 100) if self.stats['input_annotations'] > 0 else 0
        
        print(f"📊 CONVERSION STATISTICS:")
        print(f"   Input annotations: {self.stats['input_annotations']:,}")
        print(f"   Output annotations: {self.stats['total_annotations']:,}")
        print(f"   Skipped annotations: {self.stats['skipped_annotations']:,}")
        print(f"   Retention rate: {retention_rate:.1f}%")
        print(f"   Final images: {self.stats['total_images']:,}")
        print(f"   Final classes: {self.num_classes}")
        
        print(f"\\n📂 SPLIT DISTRIBUTION:")
        total_imgs = sum(self.stats['split_distribution'].values())
        for split in ['train', 'val', 'test']:
            count = self.stats['split_distribution'][split]
            percentage = count / total_imgs * 100 if total_imgs > 0 else 0
            print(f"   {split:5s}: {count:6,d} images ({percentage:4.1f}%)")
        
        print(f"\\n🏷️ CLASS DISTRIBUTION (11-class taxonomy):")
        total_annotations = sum(self.stats['class_distribution'].values())
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            count = self.stats['class_distribution'].get(class_id, 0)
            percentage = count / total_annotations * 100 if total_annotations > 0 else 0
            print(f"   {class_id:2d} {class_name:15s}: {count:6,d} ({percentage:5.1f}%)")
        
        print(f"\\n📈 DATASET CONTRIBUTIONS:")
        for dataset_name, class_counts in self.stats['dataset_contributions'].items():
            total_from_dataset = sum(class_counts.values())
            percentage = total_from_dataset / total_annotations * 100 if total_annotations > 0 else 0
            print(f"   {dataset_name:20s}: {total_from_dataset:6,d} annotations ({percentage:4.1f}%)")
    
    def merge(self):
        """Main merge process"""
        print("🚀 FINAL 11-CLASS DATASET MERGER - CORRECT APPROACH")
        print("=" * 70)
        print("🎯 Merge 4 datasets (keeping original formats) -> 11-class taxonomy")
        print("📋 Datasets:")
        print("   1. Object Detection 35 (35 classes) -> 11 traffic classes")
        print("   2. Intersection Flow 5K (8 classes) -> 6 traffic classes") 
        print("   3. VN Traffic Sign (29 classes) -> 1 traffic_sign class")
        print("   4. Road Issues (7 classes) -> 2 traffic classes")
        
        # Setup output structure
        self.setup_output_structure()
        
        # Process all datasets
        self.process_all_datasets()
        
        # Create configuration
        self.create_dataset_config()
        
        # Print summary
        self.print_final_summary()
        
        print(f"\\n✅ MERGE COMPLETE!")
        print(f"📁 Output: {self.output_root}")
        print(f"🎯 Ready for YOLOv12 training with 11 optimized classes!")

if __name__ == "__main__":
    merger = FinalCorrectMerger()
    merger.merge()