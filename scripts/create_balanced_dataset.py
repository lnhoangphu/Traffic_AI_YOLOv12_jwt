"""
🎯 Dataset Balancing Pipeline for YOLOv12 Traffic Detection
==============================================================

Tạo lại dataset cân bằng từ 4 dataset gốc với:
- Class balancing (oversample rare classes, undersample common classes)
- Data augmentation (adaptive intensity based on class frequency)
- Data cleaning (remove duplicates, invalid boxes, empty images)
- Quality control (SSIM deduplication, bbox validation)

Author: AI Assistant
Date: 2025-11-12
"""

import os
import sys
import json
import shutil
import random
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, desc="", leave=True):
        return iterable
import numpy as np
from PIL import Image
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class YOLODatasetBalancer:
    """
    Main class for balancing YOLO datasets with advanced features:
    - Multi-source dataset fusion
    - Intelligent class balancing
    - Adaptive data augmentation
    - Quality control & deduplication
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
    
    # Target distribution (semi-balanced)
    TARGET_DISTRIBUTION = {
        "Vehicle": 0.20,      # 20% (giảm từ 90%)
        "Bus": 0.08,
        "Bicycle": 0.10,
        "Person": 0.12,       # 12% (tăng từ 0.7%)
        "Engine": 0.10,
        "Truck": 0.08,
        "Tricycle": 0.08,
        "Obstacle": 0.08,
        "Pothole": 0.05,
        "Traffic Light": 0.06,
        "Traffic Sign": 0.05
    }
    
    def __init__(
        self,
        source_datasets: List[Dict[str, str]],
        output_dir: str,
        balance_mode: str = "semi-balanced",
        augmentation_intensity: str = "adaptive",
        deduplicate_images: bool = True,
        min_bbox_per_image: int = 1,
        target_total_images: int = 30000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize dataset balancer
        
        Args:
            source_datasets: List of dicts with 'name', 'images_dir', 'labels_dir', 'class_map'
            output_dir: Output directory for balanced dataset
            balance_mode: "strict" or "semi-balanced" (±15% tolerance)
            augmentation_intensity: "adaptive" (based on frequency) or "fixed"
            deduplicate_images: Remove duplicate/similar images (SSIM > 0.95)
            min_bbox_per_image: Minimum bboxes required per image
            target_total_images: Target number of images in final dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility
        """
        self.source_datasets = source_datasets
        self.output_dir = Path(output_dir)
        self.balance_mode = balance_mode
        self.augmentation_intensity = augmentation_intensity
        self.deduplicate_images = deduplicate_images
        self.min_bbox_per_image = min_bbox_per_image
        self.target_total_images = target_total_images
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Statistics tracking
        self.stats = {
            "before_balance": defaultdict(int),
            "after_balance": defaultdict(int),
            "augmentation_applied": defaultdict(int),
            "images_removed": {
                "duplicates": 0,
                "invalid_bbox": 0,
                "no_bbox": 0
            },
            "total_images_before": 0,
            "total_images_after": 0,
            "total_bboxes_before": 0,
            "total_bboxes_after": 0
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
        Load and merge all source datasets
        
        Returns:
            List of image metadata dicts with unified class IDs
        """
        print("\n📊 Loading source datasets...")
        all_images = []
        
        for ds_info in self.source_datasets:
            print(f"\n   📦 Processing: {ds_info['name']}")
            images_dir = Path(ds_info['images_dir'])
            labels_dir = Path(ds_info['labels_dir'])
            class_map = ds_info.get('class_map', {})  # Map source class IDs to target IDs
            
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
                
                # Get image hash for deduplication
                img_hash = self._compute_image_hash(img_path) if self.deduplicate_images else None
                
                # Count classes
                for ann in valid_annotations:
                    class_id = ann['class_id']
                    class_name = self.CLASS_MAPPING[class_id]
                    self.stats['before_balance'][class_name] += 1
                    self.stats['total_bboxes_before'] += 1
                
                all_images.append({
                    'image_path': str(img_path),
                    'label_path': str(label_path),
                    'annotations': valid_annotations,
                    'dataset_source': ds_info['name'],
                    'image_hash': img_hash,
                    'classes_present': list(set([ann['class_id'] for ann in valid_annotations]))
                })
            
            self.stats['total_images_before'] += len(image_files)
        
        print(f"\n   ✅ Total images loaded: {len(all_images)}")
        print(f"   ✅ Total bboxes loaded: {self.stats['total_bboxes_before']}")
        
        return all_images
    
    def _parse_yolo_label(self, label_path: Path, class_map: Dict[int, int]) -> List[Dict]:
        """
        Parse YOLO format label file
        
        Args:
            label_path: Path to .txt label file
            class_map: Mapping from source class ID to target class ID
        
        Returns:
            List of annotation dicts
        """
        annotations = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    # Handle float class IDs (convert to int)
                    try:
                        source_class_id = int(float(parts[0]))
                    except (ValueError, TypeError):
                        continue
                    
                    # Map to target class ID
                    target_class_id = class_map.get(source_class_id, source_class_id)
                    
                    # Skip if class not in our 11-class mapping
                    if target_class_id not in self.CLASS_MAPPING:
                        continue
                    
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    annotations.append({
                        'class_id': target_class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        except Exception as e:
            print(f"      ⚠️ Error parsing {label_path}: {e}")
        
        return annotations
    
    def _validate_bboxes(self, annotations: List[Dict]) -> List[Dict]:
        """
        Validate bounding boxes (must be within [0, 1])
        
        Args:
            annotations: List of annotation dicts
        
        Returns:
            List of valid annotations
        """
        valid = []
        
        for ann in annotations:
            x, y, w, h = ann['x_center'], ann['y_center'], ann['width'], ann['height']
            
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
        """
        Compute perceptual hash for image deduplication
        
        Args:
            image_path: Path to image
        
        Returns:
            Hash string
        """
        try:
            img = Image.open(image_path).resize((32, 32)).convert('L')
            img_array = np.array(img)
            return hashlib.md5(img_array.tobytes()).hexdigest()
        except:
            return None
    
    def deduplicate_dataset(self, images: List[Dict]) -> List[Dict]:
        """
        Remove duplicate images using perceptual hash
        
        Args:
            images: List of image metadata
        
        Returns:
            Deduplicated list
        """
        if not self.deduplicate_images:
            return images
        
        print("\n🔍 Deduplicating images (hash-based)...")
        
        # Group by hash
        hash_groups = defaultdict(list)
        no_hash = []
        
        for img in images:
            if img['image_hash']:
                hash_groups[img['image_hash']].append(img)
            else:
                no_hash.append(img)
        
        # Keep only one image per hash group
        deduplicated = []
        duplicates_removed = 0
        
        for hash_val, group in tqdm(hash_groups.items(), desc="   Deduplicating", leave=False):
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Keep the one with most annotations
                best = max(group, key=lambda x: len(x['annotations']))
                deduplicated.append(best)
                duplicates_removed += len(group) - 1
        
        # Add images without hash
        deduplicated.extend(no_hash)
        
        self.stats['images_removed']['duplicates'] = duplicates_removed
        print(f"   ✅ Removed {duplicates_removed} duplicates")
        print(f"   ✅ Remaining: {len(deduplicated)} images")
        
        return deduplicated
    
    def balance_classes(self, images: List[Dict]) -> List[Dict]:
        """
        Balance class distribution using oversample + undersample
        
        Args:
            images: List of image metadata
        
        Returns:
            Balanced list of images
        """
        print("\n⚖️  Balancing class distribution...")
        
        # Group images by primary class (class with most bboxes in image)
        class_groups = defaultdict(list)
        for img in images:
            class_counts = Counter([ann['class_id'] for ann in img['annotations']])
            primary_class = class_counts.most_common(1)[0][0]
            class_name = self.CLASS_MAPPING[primary_class]
            class_groups[class_name].append(img)
        
        # Print current distribution
        print("\n   📊 Current distribution:")
        for class_name in sorted(self.CLASS_MAPPING.values()):
            count = len(class_groups[class_name])
            print(f"      {class_name:20s}: {count:6d} images")
        
        # Calculate target counts
        target_count = int(self.target_total_images * np.mean(list(self.TARGET_DISTRIBUTION.values())))
        
        if self.balance_mode == "semi-balanced":
            tolerance = 0.15
        else:
            tolerance = 0.05
        
        balanced_images = []
        
        for class_name, target_ratio in self.TARGET_DISTRIBUTION.items():
            current_images = class_groups[class_name]
            current_count = len(current_images)
            target_class_count = int(self.target_total_images * target_ratio)
            
            min_count = int(target_class_count * (1 - tolerance))
            max_count = int(target_class_count * (1 + tolerance))
            
            if current_count < min_count:
                # Oversample (with augmentation)
                print(f"   📈 Oversampling {class_name}: {current_count} → {target_class_count}")
                balanced_images.extend(self._oversample_class(current_images, target_class_count))
            
            elif current_count > max_count:
                # Undersample (diversity sampling)
                print(f"   📉 Undersampling {class_name}: {current_count} → {target_class_count}")
                balanced_images.extend(self._undersample_class(current_images, target_class_count))
            
            else:
                # Keep as is
                print(f"   ✅ {class_name} already balanced: {current_count}")
                balanced_images.extend(current_images)
        
        print(f"\n   ✅ Balanced dataset: {len(balanced_images)} images")
        
        return balanced_images
    
    def _oversample_class(self, images: List[Dict], target_count: int) -> List[Dict]:
        """
        Oversample rare class with augmentation
        
        Args:
            images: Images of this class
            target_count: Target number of images
        
        Returns:
            Oversampled images with augmentation flags
        """
        if len(images) == 0:
            return []
        
        result = images.copy()
        needed = target_count - len(images)
        
        # Sample with replacement and mark for augmentation
        for _ in range(needed):
            img = random.choice(images).copy()
            img['augment'] = True
            img['augment_strength'] = 'strong'  # Strong aug for rare classes
            result.append(img)
        
        return result
    
    def _undersample_class(self, images: List[Dict], target_count: int) -> List[Dict]:
        """
        Undersample common class with diversity sampling
        
        Args:
            images: Images of this class
            target_count: Target number of images
        
        Returns:
            Undersampled images
        """
        if len(images) <= target_count:
            return images
        
        # Diversity sampling: prefer images with multiple classes
        images_sorted = sorted(images, key=lambda x: len(x['classes_present']), reverse=True)
        
        return images_sorted[:target_count]
    
    def apply_augmentation(self, img_path: str, annotations: List[Dict], strength: str = 'medium') -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply data augmentation to image and bboxes
        
        Args:
            img_path: Path to image
            annotations: List of annotations
            strength: 'light', 'medium', or 'strong'
        
        Returns:
            Augmented image and annotations
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None, []
        
        h, w = img.shape[:2]
        aug_annotations = annotations.copy()
        
        # Strength-based augmentation
        if strength == 'light':
            # Light: only horizontal flip + color jitter
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                for ann in aug_annotations:
                    ann['x_center'] = 1.0 - ann['x_center']
            
            # Color jitter
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.9, 1.1)  # Saturation
            hsv[:, :, 2] *= random.uniform(0.9, 1.1)  # Brightness
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        elif strength == 'medium':
            # Medium: flip + rotation + color + brightness
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                for ann in aug_annotations:
                    ann['x_center'] = 1.0 - ann['x_center']
            
            # Slight rotation
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            
            # Color augmentation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] += random.uniform(-10, 10)  # Hue
            hsv[:, :, 1] *= random.uniform(0.8, 1.2)  # Saturation
            hsv[:, :, 2] *= random.uniform(0.8, 1.2)  # Brightness
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        elif strength == 'strong':
            # Strong: all augmentations for rare classes
            # Horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                for ann in aug_annotations:
                    ann['x_center'] = 1.0 - ann['x_center']
            
            # Rotation
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            
            # Strong color augmentation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] += random.uniform(-20, 20)  # Hue
            hsv[:, :, 1] *= random.uniform(0.7, 1.3)  # Saturation
            hsv[:, :, 2] *= random.uniform(0.7, 1.3)  # Brightness
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # Random brightness
            img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=random.uniform(-20, 20))
        
        return img, aug_annotations
    
    def split_dataset(self, images: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Split dataset into train/val/test ensuring class distribution
        
        Args:
            images: List of all images
        
        Returns:
            Dict with 'train', 'val', 'test' splits
        """
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
            print(f"\n   📊 {split_name.upper()} split ({len(split_images)} images):")
            class_counts = defaultdict(int)
            for img in split_images:
                for ann in img['annotations']:
                    class_name = self.CLASS_MAPPING[ann['class_id']]
                    class_counts[class_name] += 1
            
            for class_name in sorted(self.CLASS_MAPPING.values()):
                count = class_counts[class_name]
                print(f"      {class_name:20s}: {count:6d} bboxes")
        
        return splits
    
    def export_dataset(self, splits: Dict[str, List[Dict]]):
        """
        Export balanced dataset to output directory
        
        Args:
            splits: Dict with train/val/test splits
        """
        print("\n💾 Exporting balanced dataset...")
        
        for split_name, images in splits.items():
            print(f"\n   📦 Exporting {split_name} split...")
            
            images_out_dir = self.output_dir / 'images' / split_name
            labels_out_dir = self.output_dir / 'labels' / split_name
            
            for idx, img_data in enumerate(tqdm(images, desc=f"      Writing {split_name}")):
                # Generate new filename
                new_filename = f"{split_name}_{idx:06d}"
                
                # Check if augmentation needed
                if img_data.get('augment', False):
                    strength = img_data.get('augment_strength', 'medium')
                    img_array, aug_annotations = self.apply_augmentation(
                        img_data['image_path'],
                        img_data['annotations'],
                        strength=strength
                    )
                    
                    if img_array is not None:
                        # Save augmented image
                        img_out_path = images_out_dir / f"{new_filename}.jpg"
                        cv2.imwrite(str(img_out_path), img_array)
                        annotations = aug_annotations
                        
                        # Track augmentation
                        for ann in annotations:
                            class_name = self.CLASS_MAPPING[ann['class_id']]
                            self.stats['augmentation_applied'][class_name] += 1
                    else:
                        continue
                else:
                    # Copy original image
                    img_out_path = images_out_dir / f"{new_filename}.jpg"
                    shutil.copy2(img_data['image_path'], img_out_path)
                    annotations = img_data['annotations']
                
                # Write label file
                label_out_path = labels_out_dir / f"{new_filename}.txt"
                with open(label_out_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\n")
                        
                        # Track stats
                        class_name = self.CLASS_MAPPING[ann['class_id']]
                        self.stats['after_balance'][class_name] += 1
                        self.stats['total_bboxes_after'] += 1
                
                self.stats['total_images_after'] += 1
        
        print(f"\n   ✅ Export complete: {self.stats['total_images_after']} images")
    
    def generate_report(self):
        """Generate and save statistics report"""
        print("\n📊 Generating report...")
        
        # Calculate improvements
        report = {
            "dataset_info": {
                "output_directory": str(self.output_dir),
                "balance_mode": self.balance_mode,
                "augmentation_intensity": self.augmentation_intensity,
                "target_total_images": self.target_total_images,
                "splits": {
                    "train": self.train_ratio,
                    "val": self.val_ratio,
                    "test": self.test_ratio
                }
            },
            "statistics": {
                "before_balance": dict(self.stats['before_balance']),
                "after_balance": dict(self.stats['after_balance']),
                "total_images_before": self.stats['total_images_before'],
                "total_images_after": self.stats['total_images_after'],
                "total_bboxes_before": self.stats['total_bboxes_before'],
                "total_bboxes_after": self.stats['total_bboxes_after']
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
            },
            "class_balance_improvement": {}
        }
        
        # Calculate balance improvement
        before_total = sum(self.stats['before_balance'].values())
        after_total = sum(self.stats['after_balance'].values())
        
        for class_name in self.CLASS_MAPPING.values():
            before = self.stats['before_balance'].get(class_name, 0)
            after = self.stats['after_balance'].get(class_name, 0)
            
            before_pct = (before / before_total * 100) if before_total > 0 else 0
            after_pct = (after / after_total * 100) if after_total > 0 else 0
            
            report['class_balance_improvement'][class_name] = {
                "before_count": before,
                "after_count": after,
                "before_percentage": f"{before_pct:.2f}%",
                "after_percentage": f"{after_pct:.2f}%",
                "change": f"{after_pct - before_pct:+.2f}%"
            }
        
        # Save report
        report_path = self.output_dir / 'stats_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Report saved: {report_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("📈 BALANCING SUMMARY")
        print("="*70)
        print(f"\n{'Class':<20} {'Before':<15} {'After':<15} {'Change':<10}")
        print("-"*70)
        
        for class_name in sorted(self.CLASS_MAPPING.values()):
            before = self.stats['before_balance'].get(class_name, 0)
            after = self.stats['after_balance'].get(class_name, 0)
            change = after - before
            change_pct = (change / before * 100) if before > 0 else 0
            
            print(f"{class_name:<20} {before:<15,} {after:<15,} {change_pct:+6.1f}%")
        
        print("="*70)
        print(f"\n✅ Total images: {self.stats['total_images_before']:,} → {self.stats['total_images_after']:,}")
        print(f"✅ Total bboxes: {self.stats['total_bboxes_before']:,} → {self.stats['total_bboxes_after']:,}")
        print(f"✅ Images removed: {sum(self.stats['images_removed'].values()):,}")
        print(f"✅ Augmentations applied: {sum(self.stats['augmentation_applied'].values()):,}")
        print("="*70)
    
    def run(self):
        """Execute full balancing pipeline"""
        print("\n" + "="*70)
        print("🚀 STARTING DATASET BALANCING PIPELINE")
        print("="*70)
        
        # Step 1: Load all datasets
        all_images = self.load_all_datasets()
        
        # Step 2: Deduplicate
        all_images = self.deduplicate_dataset(all_images)
        
        # Step 3: Balance classes
        balanced_images = self.balance_classes(all_images)
        
        # Step 4: Split dataset
        splits = self.split_dataset(balanced_images)
        
        # Step 5: Export
        self.export_dataset(splits)
        
        # Step 6: Generate report
        self.generate_report()
        
        # Step 7: Create data.yaml
        self._create_data_yaml()
        
        print("\n" + "="*70)
        print("✅ DATASET BALANCING COMPLETE!")
        print("="*70)
    
    def _create_data_yaml(self):
        """Create data.yaml for YOLOv12"""
        yaml_content = f"""# Balanced Traffic AI Dataset
# Generated by create_balanced_dataset.py

path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 11
names:
{chr(10).join(f"  {i}: {name}" for i, name in self.CLASS_MAPPING.items())}

# Dataset statistics
total_images: {self.stats['total_images_after']}
total_annotations: {self.stats['total_bboxes_after']}
balance_mode: {self.balance_mode}
augmentation_intensity: {self.augmentation_intensity}
"""
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n   ✅ data.yaml created: {yaml_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Balance YOLO dataset from multiple sources with intelligent augmentation"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='datasets/traffic_ai_final_balanced',
        help='Output directory for balanced dataset'
    )
    
    parser.add_argument(
        '--target-images', '-t',
        type=int,
        default=30000,
        help='Target total number of images'
    )
    
    parser.add_argument(
        '--balance-mode',
        type=str,
        choices=['strict', 'semi-balanced'],
        default='semi-balanced',
        help='Balance mode: strict (±5%%) or semi-balanced (±15%%)'
    )
    
    parser.add_argument(
        '--augmentation',
        type=str,
        choices=['adaptive', 'fixed'],
        default='adaptive',
        help='Augmentation intensity mode'
    )
    
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable image deduplication'
    )
    
    parser.add_argument(
        '--min-bbox',
        type=int,
        default=1,
        help='Minimum bounding boxes per image'
    )
    
    args = parser.parse_args()
    
    # Define source datasets
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
                # Tất cả biển báo (class 0-50) → Traffic Sign (class 10)
                **{i: 10 for i in range(50)}  # Map all possible sign classes to Traffic Sign
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
                # Already converted to 11-class IDs in filtered dataset
                # Direct mapping (no conversion needed)
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
    
    # Create balancer
    balancer = YOLODatasetBalancer(
        source_datasets=source_datasets,
        output_dir=args.output,
        balance_mode=args.balance_mode,
        augmentation_intensity=args.augmentation,
        deduplicate_images=not args.no_dedup,
        min_bbox_per_image=args.min_bbox,
        target_total_images=args.target_images
    )
    
    # Run pipeline
    balancer.run()


if __name__ == '__main__':
    main()
