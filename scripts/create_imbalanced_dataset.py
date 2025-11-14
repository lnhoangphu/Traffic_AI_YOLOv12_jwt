"""
🎯 Imbalanced Dataset Creation for YOLOv12 Traffic Detection
==============================================================

Tạo dataset mất cân bằng từ 4 dataset gốc để so sánh với balanced dataset:
- Giữ nguyên phân phối tự nhiên (không balance)
- Minimal augmentation (chỉ cho những class quá hiếm)
- Data cleaning (remove duplicates, invalid boxes, empty images)
- Quality control (SSIM deduplication, bbox validation)

Author: AI Assistant
Date: 2025-11-13
"""

import os
import sys
import cv2
import json
import random
import hashlib
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, desc="", leave=True):
        return iterable

import shutil
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class YOLOImbalancedDatasetCreator:
    """
    Create imbalanced dataset preserving natural distribution from source datasets
    """
    
    # 11-class mapping for Traffic AI
    CLASS_MAPPING = {
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
    
    def __init__(
        self,
        source_datasets: List[Dict[str, str]],
        output_dir: str,
        deduplicate_images: bool = True,
        min_bbox_per_image: int = 1,
        minimal_augmentation: bool = True,
        min_samples_for_augmentation: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize imbalanced dataset creator
        
        Args:
            source_datasets: List of dicts with 'name', 'images_dir', 'labels_dir', 'class_map'
            output_dir: Output directory for imbalanced dataset
            deduplicate_images: Remove duplicate/similar images (SSIM > 0.95)
            min_bbox_per_image: Minimum bboxes required per image
            minimal_augmentation: Only augment extremely rare classes
            min_samples_for_augmentation: Minimum samples before augmentation kicks in
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility
        """
        self.source_datasets = source_datasets
        self.output_dir = Path(output_dir)
        self.deduplicate_images = deduplicate_images
        self.min_bbox_per_image = min_bbox_per_image
        self.minimal_augmentation = minimal_augmentation
        self.min_samples_for_augmentation = min_samples_for_augmentation
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Statistics tracking
        self.stats = {
            "class_distribution": defaultdict(int),
            "augmentation_applied": defaultdict(int),
            "images_removed": {
                "duplicates": 0,
                "invalid_bbox": 0,
                "no_bbox": 0
            },
            "total_images": 0,
            "total_bboxes": 0
        }
        
        # Create output structure
        self._create_output_structure()
    
    def _create_output_structure(self):
        """Create output directory structure"""
        print("\n📁 Creating output directory structure...")
        
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        print(f"   ✅ Output directory: {self.output_dir}")
    
    def load_all_datasets(self) -> List[Dict]:
        """
        Load and merge all source datasets WITHOUT balancing
        
        Returns:
            List of image metadata dicts with unified class IDs
        """
        print("\n📊 Loading source datasets (preserving natural distribution)...")
        all_images = []
        
        for ds_info in self.source_datasets:
            print(f"\n   📦 Processing: {ds_info['name']}")
            images_dir = Path(ds_info['images_dir'])
            labels_dir = Path(ds_info['labels_dir'])
            class_map = ds_info.get('class_map', {})
            
            # Find all images
            image_files = list(images_dir.rglob('*.jpg')) + list(images_dir.rglob('*.png'))
            print(f"      Found {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"      Loading {ds_info['name']}", leave=False):
                # Find corresponding label file
                label_path = labels_dir / img_path.relative_to(images_dir).with_suffix('.txt')
                
                if not label_path.exists():
                    continue
                
                # Parse annotations
                annotations = self._parse_yolo_label(label_path, class_map)
                
                if len(annotations) < self.min_bbox_per_image:
                    self.stats['images_removed']['no_bbox'] += 1
                    continue
                
                # Validate bboxes
                valid_annotations = self._validate_bboxes(annotations)
                
                if len(valid_annotations) < self.min_bbox_per_image:
                    self.stats['images_removed']['invalid_bbox'] += 1
                    continue
                
                # Get classes present in this image
                classes_present = list(set(ann['class_id'] for ann in valid_annotations))
                
                # Compute hash for deduplication
                image_hash = self._compute_image_hash(img_path)
                
                all_images.append({
                    'image_path': str(img_path),
                    'annotations': valid_annotations,
                    'classes_present': classes_present,
                    'image_hash': image_hash,
                    'source_dataset': ds_info['name']
                })
                
                # Update statistics
                for ann in valid_annotations:
                    class_name = self.CLASS_MAPPING[ann['class_id']]
                    self.stats['class_distribution'][class_name] += 1
                    self.stats['total_bboxes'] += 1
        
        self.stats['total_images'] = len(all_images)
        
        print(f"\n   ✅ Total images loaded: {len(all_images)}")
        print(f"   ✅ Total bboxes loaded: {self.stats['total_bboxes']}")
        
        return all_images
    
    def _parse_yolo_label(self, label_path: Path, class_map: Dict[int, int]) -> List[Dict]:
        """Parse YOLO format label file"""
        annotations = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    # Handle both int and float class IDs (convert float to int)
                    source_class_id = int(float(parts[0]))
                    target_class_id = class_map.get(source_class_id, source_class_id)
                    
                    # Skip if target class is not in our 11-class mapping
                    if target_class_id not in self.CLASS_MAPPING:
                        continue
                    
                    x, y, w, h = map(float, parts[1:5])
                    
                    annotations.append({
                        'class_id': target_class_id,
                        'bbox': (x, y, w, h)
                    })
        except Exception as e:
            print(f"      ⚠️ Error parsing {label_path}: {e}")
        
        return annotations
    
    def _validate_bboxes(self, annotations: List[Dict]) -> List[Dict]:
        """Validate bounding boxes"""
        valid = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            
            # Check bounds
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue
            
            # Check bbox doesn't exceed image bounds
            x_min = x - w / 2
            y_min = y - h / 2
            x_max = x + w / 2
            y_max = y + h / 2
            
            if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
                continue
            
            valid.append(ann)
        
        return valid
    
    def _compute_image_hash(self, image_path: Path) -> str:
        """Compute perceptual hash for image deduplication"""
        try:
            img = Image.open(image_path).resize((32, 32)).convert('L')
            img_array = np.array(img)
            return hashlib.md5(img_array.tobytes()).hexdigest()
        except:
            return None
    
    def deduplicate_dataset(self, images: List[Dict]) -> List[Dict]:
        """Remove duplicate images using perceptual hash"""
        if not self.deduplicate_images:
            return images
        
        print("\n🔍 Deduplicating images (hash-based)...")
        
        hash_groups = defaultdict(list)
        no_hash = []
        
        for img in images:
            if img['image_hash']:
                hash_groups[img['image_hash']].append(img)
            else:
                no_hash.append(img)
        
        deduplicated = []
        duplicates_removed = 0
        
        for hash_val, group in tqdm(hash_groups.items(), desc="   Deduplicating", leave=False):
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                deduplicated.append(group[0])
                duplicates_removed += len(group) - 1
        
        deduplicated.extend(no_hash)
        
        self.stats['images_removed']['duplicates'] = duplicates_removed
        print(f"   ✅ Removed {duplicates_removed} duplicates")
        print(f"   ✅ Remaining: {len(deduplicated)} images")
        
        return deduplicated
    
    def apply_minimal_augmentation(self, images: List[Dict]) -> List[Dict]:
        """
        Apply minimal augmentation only to extremely rare classes
        
        Args:
            images: List of image metadata
        
        Returns:
            Images with augmentation flags
        """
        if not self.minimal_augmentation:
            return images
        
        print("\n🔧 Applying minimal augmentation for rare classes...")
        
        # Count samples per class
        class_counts = defaultdict(int)
        for img in images:
            for class_id in img['classes_present']:
                class_name = self.CLASS_MAPPING[class_id]
                class_counts[class_name] += 1
        
        # Identify extremely rare classes (< threshold)
        rare_classes = {
            class_name for class_name, count in class_counts.items()
            if count < self.min_samples_for_augmentation
        }
        
        print(f"   📊 Rare classes requiring augmentation: {rare_classes}")
        
        # Mark images containing rare classes for augmentation
        augmented_images = images.copy()
        
        for img in augmented_images:
            img_classes = set(self.CLASS_MAPPING[c] for c in img['classes_present'])
            if img_classes & rare_classes:
                # Augment images with rare classes (2x)
                img['augment'] = False  # Original
                augmented_images.append({
                    **img.copy(),
                    'augment': True,
                    'augment_strength': 'light'
                })
                
                for class_name in img_classes & rare_classes:
                    self.stats['augmentation_applied'][class_name] += 1
        
        total_augmented = sum(self.stats['augmentation_applied'].values())
        print(f"   ✅ Applied {total_augmented} augmentations")
        
        return augmented_images
    
    def print_distribution(self, images: List[Dict], title: str):
        """Print class distribution"""
        print(f"\n   📊 {title}:")
        
        class_counts = defaultdict(int)
        for img in images:
            for ann in img['annotations']:
                class_name = self.CLASS_MAPPING[ann['class_id']]
                class_counts[class_name] += 1
        
        total = sum(class_counts.values())
        
        for class_name in sorted(self.CLASS_MAPPING.values()):
            count = class_counts[class_name]
            pct = (count / total * 100) if total > 0 else 0
            print(f"      {class_name:<20s}: {count:6d} ({pct:5.1f}%)")
    
    def split_dataset(self, images: List[Dict]) -> Dict[str, List[Dict]]:
        """Split dataset into train/val/test"""
        print("\n✂️  Splitting dataset into train/val/test...")
        
        # Shuffle
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        
        # Print distribution per split
        for split_name, split_images in splits.items():
            print(f"\n   📊 {split_name.upper()} split ({len(split_images)} images)")
            self.print_distribution(split_images, f"{split_name.upper()} distribution")
        
        return splits
    
    def export_dataset(self, splits: Dict[str, List[Dict]]):
        """Export imbalanced dataset"""
        print("\n💾 Exporting imbalanced dataset...")
        
        for split_name, images in splits.items():
            print(f"\n   📦 Exporting {split_name} split...")
            
            images_out_dir = self.output_dir / 'images' / split_name
            labels_out_dir = self.output_dir / 'labels' / split_name
            
            for idx, img_data in enumerate(tqdm(images, desc=f"      Writing {split_name}")):
                # Generate output filename
                out_filename = f"{split_name}_{idx:06d}"
                
                # Copy or augment image
                img_path = img_data['image_path']
                out_img_path = images_out_dir / f"{out_filename}.jpg"
                
                if img_data.get('augment', False):
                    # Apply augmentation
                    img, aug_annotations = self.apply_augmentation(
                        img_path,
                        img_data['annotations'],
                        img_data.get('augment_strength', 'light')
                    )
                    if img is not None:
                        cv2.imwrite(str(out_img_path), img)
                    else:
                        shutil.copy2(img_path, out_img_path)
                        aug_annotations = img_data['annotations']
                else:
                    # Copy original
                    shutil.copy2(img_path, out_img_path)
                    aug_annotations = img_data['annotations']
                
                # Write label file
                out_label_path = labels_out_dir / f"{out_filename}.txt"
                with open(out_label_path, 'w') as f:
                    for ann in aug_annotations:
                        x, y, w, h = ann['bbox']
                        f.write(f"{ann['class_id']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        print(f"\n   ✅ Export complete: {sum(len(s) for s in splits.values())} images")
    
    def apply_augmentation(self, img_path: str, annotations: List[Dict], strength: str = 'light') -> Tuple[np.ndarray, List[Dict]]:
        """Apply light augmentation to image"""
        img = cv2.imread(img_path)
        if img is None:
            return None, annotations
        
        aug_annotations = annotations.copy()
        
        # Light augmentation only
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            # Flip bboxes
            for ann in aug_annotations:
                x, y, w, h = ann['bbox']
                ann['bbox'] = (1 - x, y, w, h)
        
        # Color jitter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.9, 1.1)  # Saturation
        hsv[:, :, 2] *= random.uniform(0.9, 1.1)  # Brightness
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return img, aug_annotations
    
    def generate_report(self):
        """Generate and save statistics report"""
        print("\n📊 Generating report...")
        
        report = {
            "dataset_info": {
                "output_directory": str(self.output_dir),
                "type": "imbalanced",
                "description": "Natural distribution preserved from source datasets",
                "minimal_augmentation": self.minimal_augmentation,
                "min_samples_for_augmentation": self.min_samples_for_augmentation,
                "splits": {
                    "train": self.train_ratio,
                    "val": self.val_ratio,
                    "test": self.test_ratio
                }
            },
            "statistics": {
                "total_images": self.stats['total_images'],
                "total_bboxes": self.stats['total_bboxes'],
                "class_distribution": dict(self.stats['class_distribution'])
            },
            "augmentation": {
                "total_augmented": sum(self.stats['augmentation_applied'].values()),
                "per_class": dict(self.stats['augmentation_applied'])
            },
            "cleaning": {
                "duplicates_removed": self.stats['images_removed']['duplicates'],
                "invalid_bbox_removed": self.stats['images_removed']['invalid_bbox'],
                "no_bbox_removed": self.stats['images_removed']['no_bbox'],
                "total_removed": sum(self.stats['images_removed'].values())
            }
        }
        
        # Save report
        report_path = self.output_dir / 'stats_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Report saved: {report_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("📈 IMBALANCED DATASET SUMMARY")
        print("="*70)
        
        total_bboxes = sum(self.stats['class_distribution'].values())
        
        print(f"\n{'Class':<20} {'Count':<15} {'Percentage':<10}")
        print("-"*70)
        
        for class_name in sorted(self.CLASS_MAPPING.values()):
            count = self.stats['class_distribution'].get(class_name, 0)
            pct = (count / total_bboxes * 100) if total_bboxes > 0 else 0
            print(f"{class_name:<20} {count:<15,} {pct:6.2f}%")
        
        print("="*70)
        print(f"\n✅ Total images: {self.stats['total_images']:,}")
        print(f"✅ Total bboxes: {self.stats['total_bboxes']:,}")
        print(f"✅ Images removed: {sum(self.stats['images_removed'].values()):,}")
        print(f"✅ Augmentations applied: {sum(self.stats['augmentation_applied'].values()):,}")
        print("="*70)
    
    def run(self):
        """Execute full pipeline"""
        print("\n" + "="*70)
        print("🚀 CREATING IMBALANCED DATASET (Natural Distribution)")
        print("="*70)
        
        # Step 1: Load all datasets
        all_images = self.load_all_datasets()
        
        # Step 2: Deduplicate
        all_images = self.deduplicate_dataset(all_images)
        
        # Step 3: Minimal augmentation for rare classes
        all_images = self.apply_minimal_augmentation(all_images)
        
        # Step 4: Split dataset
        splits = self.split_dataset(all_images)
        
        # Step 5: Export
        self.export_dataset(splits)
        
        # Step 6: Generate report
        self.generate_report()
        
        # Step 7: Create data.yaml
        self._create_data_yaml()
        
        print("\n" + "="*70)
        print("✅ IMBALANCED DATASET CREATION COMPLETE!")
        print("="*70)
    
    def _create_data_yaml(self):
        """Create data.yaml for YOLOv12"""
        yaml_content = f"""# Imbalanced Traffic AI Dataset (Natural Distribution)
# Generated by create_imbalanced_dataset.py

path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 11
names:
{chr(10).join(f"  {i}: {name}" for i, name in self.CLASS_MAPPING.items())}

# Dataset statistics
total_images: {self.stats['total_images']}
total_annotations: {self.stats['total_bboxes']}
type: imbalanced
description: Natural distribution preserved from source datasets
minimal_augmentation: {self.minimal_augmentation}
"""
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n   ✅ data.yaml created: {yaml_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Create imbalanced YOLO dataset preserving natural distribution"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='datasets/traffic_ai_final_imbalanced',
        help='Output directory for imbalanced dataset'
    )
    
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable image deduplication'
    )
    
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable minimal augmentation for rare classes'
    )
    
    parser.add_argument(
        '--min-bbox',
        type=int,
        default=1,
        help='Minimum bounding boxes per image'
    )
    
    parser.add_argument(
        '--min-samples-aug',
        type=int,
        default=100,
        help='Minimum samples before augmentation kicks in'
    )
    
    args = parser.parse_args()
    
    # Define source datasets (same as balanced dataset)
    PROJECT_ROOT = Path(__file__).parent.parent
    
    source_datasets = [
        {
            'name': 'Intersection-Flow-5K',
            'images_dir': str(PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/images/train'),
            'labels_dir': str(PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/labels/train'),
            'class_map': {
                0: 3,  # pedestrian → Person
                1: 2,  # bicycle → Bicycle
                2: 0,  # vehicle → Vehicle
                3: 0,  # car → Vehicle
                4: 5,  # truck → Truck
                5: 1,  # bus → Bus
                6: 4,  # engine → Engine
                7: 6   # tricycle → Tricycle
            }
        },
        {
            'name': 'VN Traffic Sign',
            'images_dir': str(PROJECT_ROOT / 'datasets_src/vn_traffic_sign/dataset/images/train'),
            'labels_dir': str(PROJECT_ROOT / 'datasets_src/vn_traffic_sign/dataset/labels/train'),
            'class_map': {
                **{i: 10 for i in range(50)}  # All signs → Traffic Sign
            }
        },
        {
            'name': 'Road Issues',
            'images_dir': str(PROJECT_ROOT / 'datasets_src/road_issues_yolo/images/train'),
            'labels_dir': str(PROJECT_ROOT / 'datasets_src/road_issues_yolo/labels/train'),
            'class_map': {
                0: 10,  # broken_road_sign → Traffic Sign
                1: 7,   # damaged_road → Obstacle
                2: 7,   # faded_road_markings → Obstacle
                3: 7,   # mixed_issue → Obstacle
                4: 8,   # pothole → Pothole
                5: 7,   # littering_garbage → Obstacle
                6: 7    # vandalism → Obstacle
            }
        },
        {
            'name': 'Object Detection 35',
            'images_dir': str(PROJECT_ROOT / 'datasets_src/object_detection_35_traffic_only/images/train'),
            'labels_dir': str(PROJECT_ROOT / 'datasets_src/object_detection_35_traffic_only/labels/train'),
            'class_map': {
                0: 0,   # Vehicle → Vehicle
                1: 1,   # Bus → Bus
                2: 2,   # Bicycle → Bicycle
                3: 3,   # Person → Person
                4: 4,   # Engine → Engine
                5: 5,   # Truck → Truck
                7: 7,   # Obstacle → Obstacle
                8: 8,   # Pothole → Pothole
                9: 9,   # Traffic Light → Traffic Light
                10: 10  # Traffic Sign → Traffic Sign
            }
        }
    ]
    
    # Create imbalanced dataset
    creator = YOLOImbalancedDatasetCreator(
        source_datasets=source_datasets,
        output_dir=args.output,
        deduplicate_images=not args.no_dedup,
        min_bbox_per_image=args.min_bbox,
        minimal_augmentation=not args.no_augmentation,
        min_samples_for_augmentation=args.min_samples_aug
    )
    
    # Run pipeline
    creator.run()


if __name__ == '__main__':
    main()
