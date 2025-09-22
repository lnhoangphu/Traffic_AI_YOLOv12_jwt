"""
Training Script for Balanced vs Imbalanced Dataset Comparison
Script training YOLOv12 trên cả 2 datasets để nghiên cứu ảnh hưởng của data imbalance

Usage: python scripts/train_balanced_vs_imbalanced.py
"""

import os
import yaml
import argparse
from pathlib import Path
import subprocess
import time
import json

class BalancedImbalancedTrainer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.yolo_model = "yolo12n.pt"  # YOLOv12 nano model
        
        # Dataset paths
        self.balanced_dataset = self.project_root / "datasets/traffic_ai_balanced_11class"
        self.imbalanced_dataset = self.project_root / "datasets/traffic_ai_imbalanced_11class"
        
        # Training configurations
        self.base_config = {
            'epochs': 100,
            'patience': 15,
            'batch': 16,
            'imgsz': 640,
            'device': 'cpu',  # Change to 'cuda' if GPU available
            'workers': 4,
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'save': True,
            'save_period': -1,
            'cache': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'min_memory': False
        }

    def verify_datasets(self):
        """Verify both datasets exist and are properly configured"""
        print("🔍 VERIFYING DATASETS")
        print("=" * 50)
        
        # Check balanced dataset
        if not self.balanced_dataset.exists():
            print(f"❌ Balanced dataset not found: {self.balanced_dataset}")
            return False
        
        balanced_yaml = self.balanced_dataset / "data.yaml"
        if not balanced_yaml.exists():
            print(f"❌ Balanced data.yaml not found: {balanced_yaml}")
            return False
        
        # Check imbalanced dataset
        if not self.imbalanced_dataset.exists():
            print(f"❌ Imbalanced dataset not found: {self.imbalanced_dataset}")
            return False
        
        imbalanced_yaml = self.imbalanced_dataset / "data.yaml"
        if not imbalanced_yaml.exists():
            print(f"❌ Imbalanced data.yaml not found: {imbalanced_yaml}")
            return False
        
        # Load and verify configs
        with open(balanced_yaml, 'r') as f:
            balanced_config = yaml.safe_load(f)
        
        with open(imbalanced_yaml, 'r') as f:
            imbalanced_config = yaml.safe_load(f)
        
        print(f"✅ Balanced dataset: {balanced_config['nc']} classes")
        print(f"✅ Imbalanced dataset: {imbalanced_config['nc']} classes")
        
        return True

    def create_training_configs(self):
        """Create training configuration files"""
        configs_dir = self.project_root / "training/configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Balanced dataset config
        balanced_config = self.base_config.copy()
        balanced_config.update({
            'data': str(self.balanced_dataset / "data.yaml"),
            'project': 'training/runs/balanced',
            'name': 'yolov12_balanced_experiment'
        })
        
        balanced_config_path = configs_dir / "balanced_config.yaml"
        with open(balanced_config_path, 'w') as f:
            yaml.dump(balanced_config, f, default_flow_style=False)
        
        # Imbalanced dataset config
        imbalanced_config = self.base_config.copy()
        imbalanced_config.update({
            'data': str(self.imbalanced_dataset / "data.yaml"),
            'project': 'training/runs/imbalanced',
            'name': 'yolov12_imbalanced_experiment'
        })
        
        imbalanced_config_path = configs_dir / "imbalanced_config.yaml"
        with open(imbalanced_config_path, 'w') as f:
            yaml.dump(imbalanced_config, f, default_flow_style=False)
        
        print(f"✅ Created training configs:")
        print(f"   Balanced: {balanced_config_path}")
        print(f"   Imbalanced: {imbalanced_config_path}")
        
        return balanced_config_path, imbalanced_config_path

    def train_model(self, config_path, dataset_type):
        """Train YOLO model with given configuration"""
        print(f"\\n🚀 STARTING {dataset_type.upper()} TRAINING")
        print("=" * 50)
        
        # Create training command
        cmd = [
            'python', '-m', 'ultralytics.YOLO',
            'train',
            f'model={self.yolo_model}',
            f'cfg={config_path}'
        ]
        
        print(f"Training command: {' '.join(cmd)}")
        
        # Start training
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {dataset_type} training completed successfully!")
                print(f"⏱️ Training time: {training_time:.1f} seconds")
                return True, training_time, result.stdout
            else:
                print(f"❌ {dataset_type} training failed!")
                print(f"Error: {result.stderr}")
                return False, training_time, result.stderr
                
        except Exception as e:
            training_time = time.time() - start_time
            print(f"❌ Exception during {dataset_type} training: {e}")
            return False, training_time, str(e)

    def compare_results(self):
        """Compare training results between balanced and imbalanced"""
        print(f"\\n📊 COMPARING TRAINING RESULTS")
        print("=" * 50)
        
        balanced_results_dir = self.project_root / "training/runs/balanced/yolov12_balanced_experiment"
        imbalanced_results_dir = self.project_root / "training/runs/imbalanced/yolov12_imbalanced_experiment"
        
        comparison_report = f"""
🎯 BALANCED vs IMBALANCED TRAINING COMPARISON
{'=' * 60}

📁 RESULT DIRECTORIES:
   Balanced: {balanced_results_dir}
   Imbalanced: {imbalanced_results_dir}

📋 ANALYSIS INSTRUCTIONS:
   1. Check metrics in results.csv files
   2. Compare mAP@0.5 and mAP@0.5:0.95 values
   3. Analyze per-class precision and recall
   4. Look at confusion matrices
   5. Compare training/validation loss curves

🔧 KEY METRICS TO COMPARE:
   - Overall mAP (Mean Average Precision)
   - Per-class AP (Average Precision)
   - Precision and Recall for each traffic class
   - Training convergence speed
   - Model generalization (val vs train performance)

📈 EXPECTED FINDINGS:
   - Balanced dataset may show more stable training
   - Imbalanced dataset may favor dominant classes
   - Compare performance on minority classes (Traffic Light, Obstacle)
   - Analyze if balancing improves minority class detection

🎯 RESEARCH CONCLUSIONS:
   Based on results, determine:
   1. Does data balancing improve overall performance?
   2. Which classes benefit most from balancing?
   3. Trade-offs between balanced vs natural distribution
   4. Recommendations for traffic AI deployment
"""
        
        print(comparison_report)
        
        # Save comparison report
        comparison_file = self.project_root / "training/comparison_report.txt"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(comparison_report)
        
        print(f"✅ Comparison report saved: {comparison_file}")

    def run_full_experiment(self):
        """Run complete balanced vs imbalanced experiment"""
        print("🎯 BALANCED vs IMBALANCED TRAINING EXPERIMENT")
        print("=" * 60)
        print("🔬 Research Goal: Analyze impact of data balance on traffic AI performance")
        print("=" * 60)
        
        # Step 1: Verify datasets
        if not self.verify_datasets():
            print("❌ Dataset verification failed!")
            return
        
        # Step 2: Create training configs
        balanced_config, imbalanced_config = self.create_training_configs()
        
        # Step 3: Train balanced model
        print(f"\\n🎯 PHASE 1: Training on Balanced Dataset")
        balanced_success, balanced_time, balanced_output = self.train_model(balanced_config, "balanced")
        
        # Step 4: Train imbalanced model
        print(f"\\n🎯 PHASE 2: Training on Imbalanced Dataset") 
        imbalanced_success, imbalanced_time, imbalanced_output = self.train_model(imbalanced_config, "imbalanced")
        
        # Step 5: Compare results
        self.compare_results()
        
        # Step 6: Final summary
        print(f"\\n🎉 EXPERIMENT COMPLETED!")
        print(f"   Balanced training: {'✅ Success' if balanced_success else '❌ Failed'} ({balanced_time:.1f}s)")
        print(f"   Imbalanced training: {'✅ Success' if imbalanced_success else '❌ Failed'} ({imbalanced_time:.1f}s)")
        
        if balanced_success and imbalanced_success:
            print(f"\\n📋 NEXT STEPS:")
            print(f"   1. Check training/runs/ for detailed results")
            print(f"   2. Compare metrics in results.csv files")
            print(f"   3. Analyze per-class performance differences")
            print(f"   4. Write research conclusions")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv12 on balanced vs imbalanced datasets")
    parser.add_argument('--mode', choices=['verify', 'balanced', 'imbalanced', 'full'], 
                       default='full', help='Training mode')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='cpu', help='Training device (cpu/cuda)')
    
    args = parser.parse_args()
    
    trainer = BalancedImbalancedTrainer()
    
    # Update configuration based on arguments
    trainer.base_config.update({
        'epochs': args.epochs,
        'batch': args.batch,
        'device': args.device
    })
    
    if args.mode == 'verify':
        trainer.verify_datasets()
    elif args.mode == 'balanced':
        if trainer.verify_datasets():
            balanced_config, _ = trainer.create_training_configs()
            trainer.train_model(balanced_config, "balanced")
    elif args.mode == 'imbalanced':
        if trainer.verify_datasets():
            _, imbalanced_config = trainer.create_training_configs()
            trainer.train_model(imbalanced_config, "imbalanced")
    elif args.mode == 'full':
        trainer.run_full_experiment()

if __name__ == "__main__":
    main()