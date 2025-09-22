"""
Balanced vs Imbalanced Dataset Training Comparison for Traffic Object Classification

This module implements a comprehensive comparison between training YOLOv12 models on
balanced vs imbalanced datasets for traffic object detection. Optimized for RTX 3050Ti 4GB VRAM.

Key Features:
- Smart oversampling to balance minority classes
- GPU-optimized training parameters for RTX 3050Ti
- Comprehensive evaluation with multiple export formats
- Statistical analysis and visualization of results
- Memory-efficient dataset handling

Author: Traffic AI Research Team
Date: September 2025
License: MIT

Usage:
    python scripts/train_balanced_vs_imbalanced.py
    
Dependencies:
    - ultralytics>=8.0.0 (for YOLOv12)
    - torch>=2.0.0 (CUDA support)
    - matplotlib>=3.5.0
    - pandas>=1.5.0
    - numpy>=1.21.0
"""

import yaml
import json
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

class BalancedVsImbalancedTrainer:
    """
    Trainer class for comparing balanced vs imbalanced dataset performance.
    
    This class handles the complete workflow of:
    1. Analyzing dataset imbalance
    2. Creating balanced datasets through smart oversampling
    3. Training separate models on balanced/imbalanced data
    4. Comprehensive evaluation and comparison
    5. Export models in multiple formats for deployment
    
    Optimized for RTX 3050Ti with 4GB VRAM constraints.
    
    Attributes:
        dataset_path (Path): Path to the original imbalanced dataset
        base_output (Path): Base directory for experiment outputs
        class_names (dict): Mapping of class IDs to names
        class_distribution (dict): Distribution of classes in dataset
        results (dict): Storage for training and evaluation results
    """
    
    def __init__(self, dataset_path="datasets/traffic_ai"):
        """
        Initialize the trainer with dataset path and configuration.
        
        Args:
            dataset_path (str): Path to the traffic AI dataset directory
            
        Raises:
            FileNotFoundError: If dataset statistics file is not found
            JSONDecodeError: If statistics file is corrupted
        """
        self.dataset_path = Path(dataset_path)
        self.base_output = Path("experiments/balanced_vs_imbalanced")
        
        # Load dataset statistics
        stats_file = self.dataset_path / 'statistics.json'
        if not stats_file.exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_file}")
            
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid statistics file: {e}")
        
        self.class_names = self.stats['class_names']
        self.class_distribution = self.stats['class_distribution']
        
        # Results storage for comparison
        self.results = {
            'imbalanced': {},
            'balanced': {}
        }
    
    def analyze_class_imbalance(self):
        """Phân tích mức độ mất cân bằng của dataset"""
        print("📊 ANALYZING CLASS IMBALANCE")
        print("=" * 50)
        
        distribution = self.class_distribution
        total = sum(distribution.values())
        
        print(f"Total annotations: {total}")
        print(f"\nClass distribution:")
        
        class_percentages = {}
        for class_id, count in distribution.items():
            class_id = int(class_id)
            percentage = (count / total) * 100
            class_name = self.class_names[str(class_id)]
            class_percentages[class_id] = percentage
            
            status = "🔴 UNDERREPRESENTED" if percentage < 5 else "🟡 MEDIUM" if percentage < 15 else "🟢 OVERREPRESENTED"
            print(f"   {class_id}: {class_name:<15} {count:>6} ({percentage:>5.1f}%) {status}")
        
        # Calculate imbalance ratio
        max_count = max(distribution.values())
        min_count = min(distribution.values())
        imbalance_ratio = max_count / min_count
        
        print(f"\n📈 Imbalance Analysis:")
        print(f"   Max class count: {max_count}")
        print(f"   Min class count: {min_count}")
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 10:
            print("   Status: 🔴 SEVERELY IMBALANCED")
        elif imbalance_ratio > 3:
            print("   Status: 🟡 MODERATELY IMBALANCED")
        else:
            print("   Status: 🟢 BALANCED")
        
        return class_percentages, imbalance_ratio
    
    def create_balanced_dataset(self):
        """Tạo dataset cân bằng bằng cách oversample các class ít"""
        print("\n🔄 CREATING BALANCED DATASET")
        print("=" * 50)
        
        balanced_dir = Path("datasets/traffic_ai_balanced")
        
        # Clean up existing balanced dataset if exists
        if balanced_dir.exists():
            print(f"   🧹 Cleaning up existing balanced dataset...")
            shutil.rmtree(balanced_dir)
        
        balanced_dir.mkdir(exist_ok=True)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (balanced_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (balanced_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Copy data.yaml
        shutil.copy2(self.dataset_path / 'data.yaml', balanced_dir / 'data.yaml')
        
        # Update path in data.yaml
        with open(balanced_dir / 'data.yaml', 'r') as f:
            yaml_content = f.read()
        
        yaml_content = yaml_content.replace(str(self.dataset_path), str(balanced_dir))
        
        with open(balanced_dir / 'data.yaml', 'w') as f:
            f.write(yaml_content)
        
        # Balance training set only (keep val/test unchanged)
        self.balance_split('train', balanced_dir)
        
        # Copy val and test unchanged
        for split in ['val', 'test']:
            self.copy_split_unchanged(split, balanced_dir)
        
        print(f"✅ Balanced dataset created: {balanced_dir}")
        return balanced_dir
    
    def balance_split(self, split, output_dir):
        """Balance a specific split by oversampling minority classes"""
        print(f"   ⚖️ Balancing {split} split...")
        
        source_img_dir = self.dataset_path / 'images' / split
        source_label_dir = self.dataset_path / 'labels' / split
        
        output_img_dir = output_dir / 'images' / split
        output_label_dir = output_dir / 'labels' / split
        
        # Calculate target count (use median of class counts to avoid extreme oversampling)
        class_counts = {}
        label_files = list(source_label_dir.glob('*.txt'))
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                classes_in_file = set()
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        classes_in_file.add(class_id)
                
                for class_id in classes_in_file:
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Use median as target to balance classes
        target_count = int(np.median(list(class_counts.values())))
        print(f"      Target count per class: {target_count}")
        
        # Group files by dominant class
        class_files = {i: [] for i in range(8)}  # 8 classes
        
        for label_file in label_files:
            img_file = source_img_dir / f"{label_file.stem}.jpg"
            if img_file.exists():
                # Find dominant class in this file
                with open(label_file, 'r') as f:
                    class_counts_in_file = Counter()
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            class_counts_in_file[class_id] += 1
                
                if class_counts_in_file:
                    dominant_class = class_counts_in_file.most_common(1)[0][0]
                    class_files[dominant_class].append((img_file, label_file))
        
        # Oversample each class to reach target count
        file_counter = 0
        for class_id, files in class_files.items():
            if not files:
                continue
                
            current_count = len(files)
            copies_needed = max(1, target_count // current_count)
            
            class_name = self.class_names.get(str(class_id), f"class_{class_id}")
            print(f"      Class {class_id} ({class_name}): {current_count} files -> {copies_needed} copies each")
            
            for copy_idx in range(copies_needed):
                for img_file, label_file in files:
                    new_name = f"balanced_{class_id}_{file_counter:06d}"
                    file_counter += 1
                    
                    # Copy image and label with error handling
                    try:
                        shutil.copy2(img_file, output_img_dir / f"{new_name}.jpg")
                        shutil.copy2(label_file, output_label_dir / f"{new_name}.txt")
                    except Exception as e:
                        print(f"        ⚠️ Failed to copy {img_file.name}: {e}")
                        continue
        
        print(f"      ✅ Balanced split created with {file_counter} total files")
    
    def copy_split_unchanged(self, split, output_dir):
        """Copy split without modification"""
        print(f"   📋 Copying {split} unchanged...")
        
        source_img_dir = self.dataset_path / 'images' / split
        source_label_dir = self.dataset_path / 'labels' / split
        
        output_img_dir = output_dir / 'images' / split
        output_label_dir = output_dir / 'labels' / split
        
        # Check if source directories exist
        if not source_img_dir.exists():
            print(f"   ⚠️ Warning: {source_img_dir} not found, skipping {split}")
            return
        
        # Copy all files
        img_count = 0
        for img_file in source_img_dir.glob('*'):
            if img_file.is_file():
                shutil.copy2(img_file, output_img_dir)
                img_count += 1
        
        label_count = 0
        if source_label_dir.exists():
            for label_file in source_label_dir.glob('*'):
                if label_file.is_file():
                    shutil.copy2(label_file, output_label_dir)
                    label_count += 1
        
        print(f"      ✅ Copied {img_count} images and {label_count} labels for {split}")
    
    def check_training_completed(self, weights_dir, target_epochs):
        """Check if training is already completed for given target epochs
        
        Args:
            weights_dir (Path): Directory containing checkpoint files
            target_epochs (int): Target number of epochs to complete
            
        Returns:
            bool: True if training is completed, False otherwise
        """
        if not weights_dir.exists():
            return False
            
        best_checkpoint = weights_dir / "best.pt"
        last_checkpoint = weights_dir / "last.pt"
        
        # Check best.pt first (most reliable for completion)
        if best_checkpoint.exists():
            try:
                import torch
                checkpoint = torch.load(str(best_checkpoint), map_location='cpu', weights_only=False)
                completed_epochs = checkpoint.get('epoch', 0) + 1  # Convert to 1-based counting
                
                if completed_epochs >= target_epochs:
                    print(f"   ✅ Training completed: {completed_epochs}/{target_epochs} epochs (best.pt)")
                    return True
                else:
                    print(f"   📋 Partial training found: {completed_epochs}/{target_epochs} epochs (best.pt)")
            except Exception as e:
                print(f"   ⚠️ Could not read best.pt: {e}")
        
        # Check last.pt as fallback
        if last_checkpoint.exists():
            try:
                import torch
                checkpoint = torch.load(str(last_checkpoint), map_location='cpu', weights_only=False)
                completed_epochs = checkpoint.get('epoch', 0) + 1  # Convert to 1-based counting
                
                if completed_epochs >= target_epochs:
                    print(f"   ✅ Training completed: {completed_epochs}/{target_epochs} epochs (last.pt)")
                    return True
                else:
                    print(f"   📋 Partial training found: {completed_epochs}/{target_epochs} epochs (last.pt)")
            except Exception as e:
                print(f"   ⚠️ Could not read last.pt: {e}")
        
        return False
    
    def train_model(self, dataset_path, experiment_name, epochs=50):
        """Train YOLOv12 model optimized for RTX 3050Ti
        
        Args:
            dataset_path (Path): Path to dataset with data.yaml
            experiment_name (str): Name for experiment folder
            epochs (int): Number of training epochs
            
        Returns:
            ultralytics.engine.results.Results: Training results
        """
        print(f"\n🚀 TRAINING: {experiment_name.upper()}")
        print("=" * 50)
        
        # Load YOLOv12n model (official ultralytics)
        # Use absolute path to model file in project root
        model_path = Path(__file__).parent.parent / 'yolo12n.pt'
        model = YOLO(str(model_path))
        print(f"   📋 Using model: YOLOv12n ({model_path})")
        
        # Auto-detect device (CUDA if available, otherwise CPU)
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   🖥️ Device: {device.upper()}")
        
        if device == 'cuda':
            print(f"   🎯 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            batch_size = 8  # Optimized for 4GB VRAM
            amp_enabled = True  # Mixed precision for memory savings
            cache_enabled = True  # Cache for faster training
            print(f"   🚀 RTX 3050Ti optimized: batch={batch_size}, AMP=True")
        else:
            batch_size = 4  # Smaller batch for CPU
            amp_enabled = False  # No AMP on CPU
            cache_enabled = False  # No cache on CPU
            print(f"   ⚠️ CPU fallback: batch={batch_size}, AMP=False")
        
        # Check for existing checkpoints and training completion
        weights_dir = self.base_output / experiment_name / "weights"
        last_checkpoint = weights_dir / "last.pt"
        best_checkpoint = weights_dir / "best.pt"
        
        resume_checkpoint = None
        training_completed = False
        
        # Check if training is already completed by looking at best.pt
        if best_checkpoint.exists():
            try:
                import torch
                checkpoint = torch.load(str(best_checkpoint), map_location='cpu')
                completed_epochs = checkpoint.get('epoch', 0) + 1  # Convert to 1-based counting
                
                if completed_epochs >= epochs:
                    print(f"   ✅ Training already completed ({completed_epochs}/{epochs} epochs)")
                    print(f"   📋 Using existing best model: {best_checkpoint}")
                    # Load completed model instead of training
                    model = YOLO(str(best_checkpoint))
                    
                    # Create dummy results object
                    class DummyResults:
                        def __init__(self):
                            self.results_dict = {'metrics/mAP50(B)': 0.0}
                    
                    return DummyResults()
                else:
                    print(f"   � Found partial training: {completed_epochs}/{epochs} epochs in best.pt")
            except Exception as e:
                print(f"   ⚠️ Could not read best checkpoint: {e}")
        
        # Check if we can resume from last.pt
        resume_checkpoint = None
        if last_checkpoint.exists() and not training_completed:
            try:
                import torch
                checkpoint = torch.load(str(last_checkpoint), map_location='cpu', weights_only=False)
                current_epoch = checkpoint.get('epoch', 0) + 1  # Convert to 1-based counting
                
                if current_epoch >= epochs:
                    print(f"   ✅ Training already completed ({current_epoch}/{epochs} epochs)")
                    print(f"   📋 Using existing last model: {last_checkpoint}")
                    model = YOLO(str(last_checkpoint))
                    
                    class DummyResults:
                        def __init__(self):
                            self.results_dict = {'metrics/mAP50(B)': 0.0}
                    
                    return DummyResults()
                else:
                    print(f"   🔄 Can resume from epoch {current_epoch}, target: {epochs}")
                    resume_checkpoint = str(last_checkpoint)
            except Exception as e:
                print(f"   ⚠️ Could not read last checkpoint: {e}")
                resume_checkpoint = None
        
        if resume_checkpoint:
            print(f"   🔄 Resuming training from: {resume_checkpoint}")
            model = YOLO(str(resume_checkpoint))
        else:
            print(f"   🆕 Starting fresh training")
            model = YOLO(self.model_path)
        
        # Optimized training parameters
        results = model.train(
            data=str(dataset_path / 'data.yaml'),
            epochs=epochs,
            imgsz=640,                    # Standard YOLO input size
            batch=batch_size,             # Device-optimized batch size
            device=device,                # Auto-detected device
            resume=bool(resume_checkpoint),  # Resume if checkpoint exists
            
            # Project settings
            project=str(self.base_output),
            name=experiment_name,
            exist_ok=True,
            cache=cache_enabled,          # Dynamic cache based on device
            
            # Memory optimization for RTX 3050Ti 4GB VRAM
            amp=amp_enabled,              # Automatic Mixed Precision (GPU only)
            close_mosaic=10,              # Disable mosaic last 10 epochs
            
            # Training optimization
            patience=15,                  # Early stopping patience
            save_period=10,               # Save checkpoint every 10 epochs
            val=True,                     # Enable validation
            
            # Learning rate schedule
            lr0=0.01,                     # Initial learning rate
            lrf=0.1,                      # Final learning rate factor
            momentum=0.937,               # SGD momentum
            weight_decay=0.0005,          # Weight decay
            
            # Data augmentation (conservative for stability)
            hsv_h=0.015,                  # Hue augmentation
            hsv_s=0.7,                    # Saturation augmentation  
            hsv_v=0.4,                    # Value augmentation
            degrees=10,                   # Rotation degrees
            translate=0.1,                # Translation fraction
            scale=0.5,                    # Scale gain
            shear=0.0,                    # Shear degrees
            flipud=0.0,                   # Vertical flip probability
            fliplr=0.5,                   # Horizontal flip probability
            mosaic=1.0,                   # Mosaic probability
            mixup=0.1,                    # Mixup probability
            
            # Output settings
            plots=True,                   # Generate training plots
            save=True,                    # Save final model
            save_txt=False,               # Don't save label files
            save_conf=True,               # Save confidence scores
        )
        
        print(f"✅ Training completed: {experiment_name}")
        print(f"   📊 Best mAP@50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
        print(f"   📁 Model saved: {self.base_output}/{experiment_name}/weights/best.pt")
        
        return results
    
    def evaluate_model(self, model_path, dataset_path, experiment_name):
        """Evaluate trained model with comprehensive metrics
        
        Args:
            model_path (Path): Path to trained model (.pt file)
            dataset_path (Path): Path to dataset for evaluation
            experiment_name (str): Name for evaluation folder
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        print(f"\n🔍 EVALUATING: {experiment_name.upper()}")
        print("=" * 50)
        
        # Load trained model
        model = YOLO(model_path)
        print(f"   📋 Model loaded: {model_path}")
        
        # Auto-detect device
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 16 if device == 'cuda' else 8  # Adjust batch for device
        
        # Run comprehensive evaluation
        results = model.val(
            data=str(dataset_path / 'data.yaml'),
            split='test',                 # Use test split for evaluation
            imgsz=640,                    # Match training image size
            batch=batch_size,             # Device-adjusted batch size
            device=device,                # Auto-detected device
            save_json=True,               # Save COCO JSON results
            save_txt=True,                # Save prediction labels
            save_conf=True,               # Save confidence scores
            project=str(self.base_output),
            name=f"{experiment_name}_eval",
            exist_ok=True,
            plots=True,                   # Generate evaluation plots
        )
        
        # Extract comprehensive metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.p.mean() if results.box.p is not None else 0,
            'recall': results.box.r.mean() if results.box.r is not None else 0,
            'f1_score': 2 * (results.box.p.mean() * results.box.r.mean()) / (results.box.p.mean() + results.box.r.mean()) if results.box.p is not None and results.box.r is not None else 0,
        }
        
        # Per-class metrics
        if results.box.ap is not None:
            per_class_metrics = {}
            for i, class_name in self.class_names.items():
                if int(i) < len(results.box.ap):
                    per_class_metrics[class_name] = {
                        'AP50': results.box.ap[int(i)],
                        'precision': results.box.p[int(i)] if results.box.p is not None else 0,
                        'recall': results.box.r[int(i)] if results.box.r is not None else 0,
                    }
            metrics['per_class'] = per_class_metrics
        
        print(f"   📊 mAP@50: {metrics['mAP50']:.3f}")
        print(f"   📊 mAP@50-95: {metrics['mAP50-95']:.3f}")
        print(f"   📊 Precision: {metrics['precision']:.3f}")
        print(f"   📊 Recall: {metrics['recall']:.3f}")
        print(f"   📊 F1-Score: {metrics['f1_score']:.3f}")
        
        return metrics
    
    def export_models(self, imbalanced_model_path, balanced_model_path):
        """Export trained models in multiple formats for deployment
        
        Args:
            imbalanced_model_path (Path): Path to imbalanced model
            balanced_model_path (Path): Path to balanced model
        """
        print("\n📦 EXPORTING MODELS")
        print("=" * 50)
        
        export_formats = ['onnx', 'torchscript', 'tflite']
        
        for model_name, model_path in [('imbalanced', imbalanced_model_path), ('balanced', balanced_model_path)]:
            if not model_path.exists():
                print(f"   ⚠️ Model not found: {model_path}")
                continue
                
            print(f"   📋 Exporting {model_name} model...")
            model = YOLO(model_path)
            
            for fmt in export_formats:
                try:
                    exported_path = model.export(format=fmt, optimize=True)
                    print(f"      ✅ {fmt.upper()}: {exported_path}")
                except Exception as e:
                    print(f"      ❌ {fmt.upper()}: {e}")
    
    def create_comparison_plots(self):
        """Create comprehensive comparison plots and analysis"""
        print("\n📊 CREATING COMPARISON PLOTS")
        print("=" * 50)
        
        if not self.results['imbalanced'] or not self.results['balanced']:
            print("   ⚠️ Both models must be trained and evaluated first")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Balanced vs Imbalanced Dataset Training Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics for plotting
        imbalanced_metrics = self.results['imbalanced']['metrics']
        balanced_metrics = self.results['balanced']['metrics']
        
        # Overall performance comparison
        metrics_names = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1_score']
        imbalanced_values = [imbalanced_metrics[m] for m in metrics_names]
        balanced_values = [balanced_metrics[m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, imbalanced_values, width, label='Imbalanced', alpha=0.8, color='coral')
        axes[0, 0].bar(x + width/2, balanced_values, width, label='Balanced', alpha=0.8, color='lightblue')
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Overall Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Per-class mAP comparison
        if 'per_class' in imbalanced_metrics and 'per_class' in balanced_metrics:
            class_names = list(self.class_names.values())
            imbalanced_ap = [imbalanced_metrics['per_class'][name]['AP50'] for name in class_names if name in imbalanced_metrics['per_class']]
            balanced_ap = [balanced_metrics['per_class'][name]['AP50'] for name in class_names if name in balanced_metrics['per_class']]
            
            x = np.arange(len(class_names))
            axes[0, 1].bar(x - width/2, imbalanced_ap, width, label='Imbalanced', alpha=0.8, color='coral')
            axes[0, 1].bar(x + width/2, balanced_ap, width, label='Balanced', alpha=0.8, color='lightblue')
            axes[0, 1].set_xlabel('Classes')
            axes[0, 1].set_ylabel('AP@50')
            axes[0, 1].set_title('Per-Class Average Precision')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(class_names, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Class distribution visualization
        distribution = self.class_distribution
        class_names = [self.class_names[str(i)] for i in range(len(self.class_names))]
        counts = [distribution.get(str(i), 0) for i in range(len(self.class_names))]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        axes[0, 2].pie(counts, labels=class_names, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 2].set_title('Original Dataset Class Distribution')
        
        # Improvement analysis
        improvements = {}
        for metric in metrics_names:
            imbalanced_val = imbalanced_metrics[metric]
            balanced_val = balanced_metrics[metric]
            improvement = ((balanced_val - imbalanced_val) / imbalanced_val * 100) if imbalanced_val > 0 else 0
            improvements[metric] = improvement
        
        improvement_values = list(improvements.values())
        colors = ['green' if x > 0 else 'red' for x in improvement_values]
        
        axes[1, 0].bar(metrics_names, improvement_values, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].set_title('Performance Improvement (Balanced vs Imbalanced)')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45)
        
        # Training convergence comparison (if training history available)
        # This would require storing training history during training
        axes[1, 1].text(0.5, 0.5, 'Training History\n(Requires training logs)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Convergence')
        
        # Summary statistics
        summary_text = f"""
EXPERIMENT SUMMARY
===================

🔴 IMBALANCED DATASET:
   • mAP@50: {imbalanced_metrics['mAP50']:.3f}
   • mAP@50-95: {imbalanced_metrics['mAP50-95']:.3f}
   • Precision: {imbalanced_metrics['precision']:.3f}
   • Recall: {imbalanced_metrics['recall']:.3f}
   • F1-Score: {imbalanced_metrics['f1_score']:.3f}

🟢 BALANCED DATASET:
   • mAP@50: {balanced_metrics['mAP50']:.3f}
   • mAP@50-95: {balanced_metrics['mAP50-95']:.3f}
   • Precision: {balanced_metrics['precision']:.3f}
   • Recall: {balanced_metrics['recall']:.3f}
   • F1-Score: {balanced_metrics['f1_score']:.3f}

📈 BEST IMPROVEMENTS:
"""
        
        # Find best improvements
        sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
        for metric, improvement in sorted_improvements[:3]:
            summary_text += f"   • {metric}: {improvement:+.1f}%\n"
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plots
        plots_dir = self.base_output / 'plots'
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / 'balanced_vs_imbalanced_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'balanced_vs_imbalanced_comparison.pdf', bbox_inches='tight')
        
        print(f"   📊 Comparison plots saved: {plots_dir}")
        print(f"      • PNG: balanced_vs_imbalanced_comparison.png")
        print(f"      • PDF: balanced_vs_imbalanced_comparison.pdf")
        
        return fig
    
    def run_experiment(self, epochs_imbalanced=30, epochs_balanced=30):
        """Run complete balanced vs imbalanced experiment
        
        Args:
            epochs_imbalanced (int): Training epochs for imbalanced model
            epochs_balanced (int): Training epochs for balanced model
        """
        print("🚀 BALANCED VS IMBALANCED DATASET COMPARISON")
        print("=" * 60)
        
        # Step 1: Analyze class imbalance
        class_percentages, imbalance_ratio = self.analyze_class_imbalance()
        
        # Step 2: Use existing balanced dataset or create if not exists
        balanced_dataset_path = Path("datasets/traffic_ai_balanced")
        if not balanced_dataset_path.exists():
            print("\n🔄 CREATING BALANCED DATASET")
            print("=" * 50)
            balanced_dataset_path = self.create_balanced_dataset()
        else:
            print(f"\n✅ Using existing balanced dataset: {balanced_dataset_path}")
            print("=" * 50)
        
        # Step 3: Check training status and train imbalanced model
        print(f"\n{'='*60}")
        print("🔴 PHASE 1: TRAINING ON IMBALANCED DATASET")
        print("=" * 60)
        
        # Check if Phase 1 is already completed
        imbalanced_weights_dir = self.base_output / "imbalanced" / "weights"
        imbalanced_completed = self.check_training_completed(imbalanced_weights_dir, epochs_imbalanced)
        
        if imbalanced_completed:
            print("   ✅ Phase 1 already completed - skipping to Phase 2")
            imbalanced_results = None  # Will be handled in evaluation
        else:
            imbalanced_results = self.train_model(
                dataset_path=self.dataset_path,
                experiment_name="imbalanced",
                epochs=epochs_imbalanced
            )
        
        # Step 4: Train balanced model
        print(f"\n{'='*60}")
        print("🟢 PHASE 2: TRAINING ON BALANCED DATASET")
        print("=" * 60)
        
        # Check if Phase 2 is already completed
        balanced_weights_dir = self.base_output / "balanced" / "weights"
        balanced_completed = self.check_training_completed(balanced_weights_dir, epochs_balanced)
        
        if balanced_completed:
            print("   ✅ Phase 2 already completed - proceeding to evaluation")
            balanced_results = None  # Will be handled in evaluation
        else:
            balanced_results = self.train_model(
                dataset_path=balanced_dataset_path,
                experiment_name="balanced",
                epochs=epochs_balanced
            )
        
        # Step 5: Evaluate both models
        print(f"\n{'='*60}")
        print("🔍 PHASE 3: COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        imbalanced_model_path = self.base_output / "imbalanced" / "weights" / "best.pt"
        balanced_model_path = self.base_output / "balanced" / "weights" / "best.pt"
        
        imbalanced_metrics = self.evaluate_model(
            model_path=imbalanced_model_path,
            dataset_path=self.dataset_path,
            experiment_name="imbalanced"
        )
        
        balanced_metrics = self.evaluate_model(
            model_path=balanced_model_path,
            dataset_path=self.dataset_path,  # Use original test set for fair comparison
            experiment_name="balanced"
        )
        
        # Store results
        self.results['imbalanced'] = {
            'training': imbalanced_results,
            'metrics': imbalanced_metrics
        }
        self.results['balanced'] = {
            'training': balanced_results,
            'metrics': balanced_metrics
        }
        
        # Step 6: Export models in multiple formats
        print(f"\n{'='*60}")
        print("📦 PHASE 4: MODEL EXPORT")
        print("=" * 60)
        
        self.export_models(imbalanced_model_path, balanced_model_path)
        
        # Step 7: Create comprehensive comparison plots
        print(f"\n{'='*60}")
        print("📊 PHASE 5: ANALYSIS & VISUALIZATION")
        print("=" * 60)
        
        self.create_comparison_plots()
        
        # Step 8: Final summary
        print(f"\n{'='*60}")
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📋 EXPERIMENT SUMMARY:")
        print(f"   🔴 Imbalanced Model:")
        print(f"      • mAP@50: {imbalanced_metrics['mAP50']:.3f}")
        print(f"      • mAP@50-95: {imbalanced_metrics['mAP50-95']:.3f}")
        print(f"      • F1-Score: {imbalanced_metrics['f1_score']:.3f}")
        
        print(f"   🟢 Balanced Model:")
        print(f"      • mAP@50: {balanced_metrics['mAP50']:.3f}")
        print(f"      • mAP@50-95: {balanced_metrics['mAP50-95']:.3f}")
        print(f"      • F1-Score: {balanced_metrics['f1_score']:.3f}")
        
        # Calculate improvements
        map50_improvement = ((balanced_metrics['mAP50'] - imbalanced_metrics['mAP50']) / imbalanced_metrics['mAP50'] * 100) if imbalanced_metrics['mAP50'] > 0 else 0
        map95_improvement = ((balanced_metrics['mAP50-95'] - imbalanced_metrics['mAP50-95']) / imbalanced_metrics['mAP50-95'] * 100) if imbalanced_metrics['mAP50-95'] > 0 else 0
        f1_improvement = ((balanced_metrics['f1_score'] - imbalanced_metrics['f1_score']) / imbalanced_metrics['f1_score'] * 100) if imbalanced_metrics['f1_score'] > 0 else 0
        
        print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
        print(f"   • mAP@50: {map50_improvement:+.1f}%")
        print(f"   • mAP@50-95: {map95_improvement:+.1f}%")
        print(f"   • F1-Score: {f1_improvement:+.1f}%")
        
        print(f"\n📁 OUTPUT DIRECTORIES:")
        print(f"   • Experiments: {self.base_output}")
        print(f"   • Balanced Dataset: {balanced_dataset_path}")
        print(f"   • Comparison Plots: {self.base_output}/plots")
        
        return self.results

# Main execution
if __name__ == "__main__":
    # Initialize trainer
    trainer = BalancedVsImbalancedTrainer("datasets/traffic_ai")
    
    # Run complete experiment
    results = trainer.run_experiment(
        epochs_imbalanced=30,  # Adjust epochs based on your time constraints
        epochs_balanced=30     # Both models get equal training time
    )
    
    print("\n🎯 Experiment completed! Check the results in experiments/balanced_vs_imbalanced/")