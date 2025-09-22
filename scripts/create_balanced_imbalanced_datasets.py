"""
Create Balanced vs Imbalanced Datasets for Research
Tạo 2 datasets để nghiên cứu ảnh hưởng của data imbalance:
1. Balanced Dataset - Cân bằng số lượng samples cho mỗi class
2. Imbalanced Dataset - Giữ nguyên phân phối tự nhiên của data

Usage: python scripts/create_balanced_imbalanced_datasets.py
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
import math

class BalancedImbalancedDatasetCreator:
    def __init__(self):
        # Load taxonomy config
        self.load_taxonomy_config()
        
        # Dataset paths
        self.dataset_paths = {
            'object_detection_35': Path("datasets_src/object_detection_35_organized"),
            'intersection_flow_5k': Path("datasets_src/intersection_flow_5k/Intersection-Flow-5K"),
            'vn_traffic_sign': Path("datasets_src/vn_traffic_sign/dataset"),
            'road_issues': Path("datasets_src/road_issues_yolo")
        }
        
        # Output paths
        self.balanced_root = Path("datasets/traffic_ai_balanced_11class")
        self.imbalanced_root = Path("datasets/traffic_ai_imbalanced_11class")
        
        # Statistics
        self.class_samples = defaultdict(list)  # class_id -> list of (img_path, label_path, dataset_name)
        self.stats = {
            'collected_samples': Counter(),
            'balanced_final': Counter(),
            'imbalanced_final': Counter(),
            'dataset_contributions': defaultdict(lambda: defaultdict(int))
        }
        
        # Configuration for balanced dataset
        self.target_samples_per_class = 500  # Target number per class for balanced dataset
        self.min_samples_threshold = 50     # Minimum samples required for a class
        
        # Split ratios
        self.split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}

    def load_taxonomy_config(self):
        """Load 11-class taxonomy configuration"""
        config_path = Path("config/taxonomy_complete_11class.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['classes']
        self.num_classes = len(self.class_names)
        
        print(f"✅ Loaded taxonomy: {self.num_classes} classes")
        for i, name in enumerate(self.class_names):
            print(f"   {i}: {name}")

    def convert_annotation_line(self, line, dataset_name):
        """Convert annotation line using dataset-specific mapping"""
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        
        try:
            old_class_id = int(float(parts[0]))
        except ValueError:
            return None
            
        bbox_info = parts[1:5]
        
        # Get mapping for this dataset
        if dataset_name not in self.config['mapping']:
            return None
        
        dataset_mapping = self.config['mapping'][dataset_name]
        
        # Handle special case for vn_traffic_sign
        if dataset_name == 'vn_traffic_sign':
            new_class_id = 10  # Traffic Sign
        elif old_class_id in dataset_mapping:
            new_class_id = dataset_mapping[old_class_id]
        else:
            return None
        
        return f"{new_class_id} {' '.join(bbox_info)}", new_class_id

    def collect_all_samples(self):
        """Collect all samples from all datasets and organize by class"""
        print("\n🔍 COLLECTING ALL SAMPLES BY CLASS")
        print("=" * 50)
        
        for dataset_name, dataset_path in self.dataset_paths.items():
            if not dataset_path.exists():
                print(f"⚠️ Dataset not found: {dataset_path}")
                continue
                
            print(f"\n📂 Processing {dataset_name}...")
            files = self.find_dataset_files(dataset_name, dataset_path)
            print(f"   Found {len(files)} image-label pairs")
            
            for file_info in tqdm(files, desc=f"Analyzing {dataset_name}"):
                self.analyze_file_classes(
                    file_info['img_path'], 
                    file_info['label_path'], 
                    dataset_name
                )
        
        # Print collection summary
        self.print_collection_summary()

    def find_dataset_files(self, dataset_name, dataset_path):
        """Find all image-label pairs in dataset"""
        files = []
        
        if dataset_name == 'object_detection_35':
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
        
        elif dataset_name == 'intersection_flow_5k':
            # Find structure dynamically
            labels_dir = dataset_path / 'labels'
            if labels_dir.exists():
                for split in ['train', 'val', 'test']:
                    split_dir = labels_dir / split
                    if split_dir.exists():
                        for label_file in split_dir.glob('*.txt'):
                            img_file = dataset_path / 'images' / split / f"{label_file.stem}.jpg"
                            if img_file.exists():
                                files.append({
                                    'img_path': img_file,
                                    'label_path': label_file,
                                    'original_split': split
                                })
        
        elif dataset_name == 'vn_traffic_sign':
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

    def analyze_file_classes(self, img_path, label_path, dataset_name):
        """Analyze what classes are in this file and add to collection"""
        try:
            with open(label_path, 'r') as f:
                classes_in_file = set()
                
                for line in f:
                    result = self.convert_annotation_line(line, dataset_name)
                    if result:
                        converted_line, new_class_id = result
                        classes_in_file.add(new_class_id)
                
                # Add this file to each class it contains
                for class_id in classes_in_file:
                    self.class_samples[class_id].append({
                        'img_path': img_path,
                        'label_path': label_path,
                        'dataset_name': dataset_name,
                        'classes_in_file': classes_in_file
                    })
                    self.stats['collected_samples'][class_id] += 1
                    self.stats['dataset_contributions'][dataset_name][class_id] += 1
                        
        except Exception as e:
            print(f"⚠️ Error analyzing {label_path}: {e}")

    def print_collection_summary(self):
        """Print summary of collected samples"""
        print(f"\n📊 COLLECTION SUMMARY")
        print("=" * 50)
        
        for class_id in sorted(self.class_samples.keys()):
            class_name = self.class_names[class_id]
            count = len(self.class_samples[class_id])
            print(f"   {class_id}: {class_name:<15} = {count:>5} samples")
        
        print(f"\n🔗 DATASET CONTRIBUTIONS:")
        for dataset, classes in self.stats['dataset_contributions'].items():
            print(f"   {dataset}:")
            for class_id, count in sorted(classes.items()):
                class_name = self.class_names[class_id]
                print(f"      {class_id}: {class_name} = {count} samples")

    def create_balanced_dataset(self):
        """Create balanced dataset with equal samples per class"""
        print(f"\n⚖️ CREATING BALANCED DATASET")
        print("=" * 50)
        
        # Determine target count (use minimum available samples, capped at target)
        available_counts = [len(samples) for samples in self.class_samples.values()]
        min_available = min(available_counts) if available_counts else 0
        target_count = min(self.target_samples_per_class, min_available)
        
        if target_count < self.min_samples_threshold:
            print(f"⚠️ Warning: Only {target_count} samples available per class (minimum threshold: {self.min_samples_threshold})")
        
        print(f"🎯 Target samples per class: {target_count}")
        
        # Clean and create balanced output directory
        if self.balanced_root.exists():
            shutil.rmtree(self.balanced_root)
        
        for split in ['train', 'val', 'test']:
            (self.balanced_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.balanced_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Sample equal numbers from each class
        balanced_samples = []
        
        for class_id, samples in self.class_samples.items():
            # Randomly sample target_count samples from this class
            random.seed(42)  # For reproducibility
            sampled = random.sample(samples, min(target_count, len(samples)))
            balanced_samples.extend(sampled)
            self.stats['balanced_final'][class_id] = len(sampled)
        
        # Shuffle and redistribute across splits
        random.shuffle(balanced_samples)
        
        print(f"📦 Processing {len(balanced_samples)} balanced samples...")
        self.process_samples(balanced_samples, self.balanced_root, "balanced")
        
        # Create data.yaml for balanced dataset
        self.create_dataset_yaml(self.balanced_root, "Balanced Dataset")

    def create_imbalanced_dataset(self):
        """Create imbalanced dataset with natural distribution"""
        print(f"\n📊 CREATING IMBALANCED DATASET")
        print("=" * 50)
        
        # Clean and create imbalanced output directory
        if self.imbalanced_root.exists():
            shutil.rmtree(self.imbalanced_root)
        
        for split in ['train', 'val', 'test']:
            (self.imbalanced_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.imbalanced_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Use all available samples (natural imbalance)
        imbalanced_samples = []
        
        for class_id, samples in self.class_samples.items():
            imbalanced_samples.extend(samples)
            self.stats['imbalanced_final'][class_id] = len(samples)
        
        # Remove duplicates (files that contain multiple classes)
        unique_samples = {}
        for sample in imbalanced_samples:
            file_key = str(sample['img_path'])
            if file_key not in unique_samples:
                unique_samples[file_key] = sample
        
        imbalanced_samples = list(unique_samples.values())
        
        # Shuffle and redistribute
        random.seed(42)
        random.shuffle(imbalanced_samples)
        
        print(f"📦 Processing {len(imbalanced_samples)} imbalanced samples...")
        self.process_samples(imbalanced_samples, self.imbalanced_root, "imbalanced")
        
        # Create data.yaml for imbalanced dataset
        self.create_dataset_yaml(self.imbalanced_root, "Imbalanced Dataset")

    def process_samples(self, samples, output_root, dataset_type):
        """Process and copy samples to output directory"""
        
        # Calculate split indices
        n_total = len(samples)
        n_train = int(n_total * self.split_ratios['train'])
        n_val = int(n_total * self.split_ratios['val'])
        
        split_assignments = []
        for i in range(n_total):
            if i < n_train:
                split_assignments.append('train')
            elif i < n_train + n_val:
                split_assignments.append('val')
            else:
                split_assignments.append('test')
        
        # Process each sample
        file_index = 0
        split_counts = Counter()
        
        for sample, target_split in tqdm(zip(samples, split_assignments), total=len(samples), desc=f"Processing {dataset_type}"):
            success = self.process_single_sample(sample, target_split, output_root, file_index, dataset_type)
            if success:
                split_counts[target_split] += 1
                file_index += 1
        
        # Print split distribution
        print(f"   📂 Split distribution:")
        for split in ['train', 'val', 'test']:
            print(f"      {split}: {split_counts[split]} images")

    def process_single_sample(self, sample, target_split, output_root, file_index, dataset_type):
        """Process single sample file"""
        try:
            img_path = sample['img_path']
            label_path = sample['label_path']
            dataset_name = sample['dataset_name']
            
            # Read and convert annotations
            converted_annotations = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    result = self.convert_annotation_line(line, dataset_name)
                    if result:
                        converted_line, _ = result
                        converted_annotations.append(converted_line)
            
            if not converted_annotations:
                return False
            
            # Create output filenames
            dataset_prefix = {
                'object_detection_35': 'od35',
                'intersection_flow_5k': 'inter',
                'vn_traffic_sign': 'sign',
                'road_issues': 'road'
            }[dataset_name]
            
            new_name = f"{dataset_type}_{dataset_prefix}_{file_index:06d}"
            
            # Copy image
            output_img_path = output_root / 'images' / target_split / f"{new_name}.jpg"
            shutil.copy2(img_path, output_img_path)
            
            # Write converted annotations
            output_label_path = output_root / 'labels' / target_split / f"{new_name}.txt"
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(converted_annotations))
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error processing {sample['img_path']}: {e}")
            return False

    def create_dataset_yaml(self, output_root, description):
        """Create data.yaml for dataset"""
        data_yaml = {
            'path': str(output_root.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': self.num_classes,
            'names': self.class_names,
            'description': description
        }
        
        with open(output_root / 'data.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    def create_final_report(self):
        """Create comprehensive final report"""
        print(f"\n📋 FINAL COMPARISON REPORT")
        print("=" * 60)
        
        report_content = f"""
🎯 BALANCED vs IMBALANCED DATASETS COMPARISON
{'=' * 60}

📊 CLASS DISTRIBUTION COMPARISON:
{'Class':<3} {'Name':<15} {'Collected':<10} {'Balanced':<10} {'Imbalanced':<12} {'Balance Ratio':<12}
{'-' * 70}
"""
        
        for class_id in sorted(self.class_samples.keys()):
            class_name = self.class_names[class_id]
            collected = self.stats['collected_samples'][class_id]
            balanced = self.stats['balanced_final'][class_id]
            imbalanced = self.stats['imbalanced_final'][class_id]
            ratio = f"{balanced/imbalanced:.3f}" if imbalanced > 0 else "0.000"
            
            report_content += f"{class_id:<3} {class_name:<15} {collected:<10} {balanced:<10} {imbalanced:<12} {ratio:<12}\n"
        
        # Calculate imbalance metrics
        balanced_values = list(self.stats['balanced_final'].values())
        imbalanced_values = list(self.stats['imbalanced_final'].values())
        
        balanced_std = np.std(balanced_values) if balanced_values else 0
        balanced_cv = balanced_std / np.mean(balanced_values) if balanced_values and np.mean(balanced_values) > 0 else 0
        
        imbalanced_std = np.std(imbalanced_values) if imbalanced_values else 0
        imbalanced_cv = imbalanced_std / np.mean(imbalanced_values) if imbalanced_values and np.mean(imbalanced_values) > 0 else 0
        
        report_content += f"""
📈 IMBALANCE METRICS:
   Balanced Dataset:
      Standard Deviation: {balanced_std:.2f}
      Coefficient of Variation: {balanced_cv:.3f}
      
   Imbalanced Dataset:
      Standard Deviation: {imbalanced_std:.2f}
      Coefficient of Variation: {imbalanced_cv:.3f}

🎯 RESEARCH IMPLICATIONS:
   - Balanced dataset có thể cho kết quả training ổn định hơn
   - Imbalanced dataset phản ánh thực tế phân phối data
   - So sánh 2 datasets sẽ cho thấy ảnh hưởng của data imbalance
   
📂 OUTPUT DIRECTORIES:
   Balanced: {self.balanced_root}
   Imbalanced: {self.imbalanced_root}

🔧 NEXT STEPS:
   1. Train YOLOv12 trên balanced dataset
   2. Train YOLOv12 trên imbalanced dataset  
   3. So sánh performance metrics (mAP, precision, recall cho từng class)
   4. Phân tích ảnh hưởng của data imbalance lên model performance
"""
        
        print(report_content)
        
        # Save reports to files
        with open(self.balanced_root / 'dataset_info.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        with open(self.imbalanced_root / 'dataset_info.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save detailed statistics as YAML
        detailed_stats = {
            'collection_summary': {
                'total_classes': len(self.class_samples),
                'collected_samples': dict(self.stats['collected_samples']),
                'dataset_contributions': dict(self.stats['dataset_contributions'])
            },
            'balanced_dataset': {
                'samples_per_class': dict(self.stats['balanced_final']),
                'total_samples': sum(self.stats['balanced_final'].values()),
                'std_deviation': float(balanced_std),
                'coefficient_variation': float(balanced_cv)
            },
            'imbalanced_dataset': {
                'samples_per_class': dict(self.stats['imbalanced_final']),
                'total_samples': sum(self.stats['imbalanced_final'].values()),
                'std_deviation': float(imbalanced_std),
                'coefficient_variation': float(imbalanced_cv)
            },
            'class_names': self.class_names
        }
        
        with open(self.balanced_root / 'statistics.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(detailed_stats, f, default_flow_style=False, allow_unicode=True)
        
        with open(self.imbalanced_root / 'statistics.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(detailed_stats, f, default_flow_style=False, allow_unicode=True)

    def run(self):
        """Main execution function"""
        print("🚀 CREATING BALANCED vs IMBALANCED DATASETS")
        print("=" * 60)
        print("📝 Research Purpose: Analyze impact of data imbalance on model performance")
        print("=" * 60)
        
        # Step 1: Collect all samples
        self.collect_all_samples()
        
        # Step 2: Create balanced dataset
        self.create_balanced_dataset()
        
        # Step 3: Create imbalanced dataset
        self.create_imbalanced_dataset()
        
        # Step 4: Create final report
        self.create_final_report()
        
        print(f"\n🎉 DATASETS CREATION COMPLETED!")
        print(f"   ⚖️ Balanced: {self.balanced_root}")
        print(f"   📊 Imbalanced: {self.imbalanced_root}")

if __name__ == "__main__":
    creator = BalancedImbalancedDatasetCreator()
    creator.run()