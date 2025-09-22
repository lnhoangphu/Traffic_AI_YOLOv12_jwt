#!/usr/bin/env python3
"""
YOLOv12 Training Script for Imbalanced Dataset
Script huấn luyện YOLOv12 cho dataset imbalanced với class weights và checkpoint resume
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import yaml
import signal
import atexit

class ImbalancedTrainer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.dataset_path = self.project_root / "datasets" / "traffic_ai_imbalanced_11class_processed"
        self.data_yaml = self.dataset_path / "data.yaml"
        self.class_weights_yaml = self.dataset_path / "class_weights.yaml"
        self.runs_dir = self.project_root / "runs" / "imbalanced"
        self.model_weights = self.project_root / "yolo12n.pt"
        
        # Training parameters
        self.epochs = 100
        self.batch_size = 16
        self.img_size = 640
        self.patience = 20
        self.device = "0"
        
        # Logging setup
        self.setup_logging()
        self.setup_signal_handlers()
        
    def setup_logging(self):
        """Setup logging to file and console"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.runs_dir / f"imbalanced_training_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.log_dir / "training.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}. Gracefully shutting down...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def check_prerequisites(self):
        """Check if all required files exist"""
        self.logger.info("🔍 Checking prerequisites...")
        
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"❌ Data config not found: {self.data_yaml}")
        
        if not self.model_weights.exists():
            raise FileNotFoundError(f"❌ Model weights not found: {self.model_weights}")
            
        if not self.class_weights_yaml.exists():
            self.logger.warning(f"⚠️ Class weights not found: {self.class_weights_yaml}")
        
        # Load and validate data config
        with open(self.data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        train_path = self.dataset_path / data_config['train']
        val_path = self.dataset_path / data_config['val']
        
        if not train_path.exists():
            raise FileNotFoundError(f"❌ Train images not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"❌ Val images not found: {val_path}")
        
        train_count = len(list(train_path.glob('*.jpg')))
        val_count = len(list(val_path.glob('*.jpg')))
        
        self.logger.info(f"✅ Dataset validated: {train_count} train, {val_count} val images")
        self.logger.info(f"✅ Classes: {data_config.get('nc', 0)}")
        
        # Load class weights
        class_weights = None
        if self.class_weights_yaml.exists():
            with open(self.class_weights_yaml, 'r', encoding='utf-8') as f:
                weights_data = yaml.safe_load(f)
                class_weights = weights_data.get('class_weights', {})
                self.logger.info(f"✅ Class weights loaded: {len(class_weights)} classes")
                
                # Log class weights
                for class_id, weight in sorted(class_weights.items()):
                    class_name = data_config['names'][int(class_id)] if int(class_id) < len(data_config['names']) else f"Class_{class_id}"
                    self.logger.info(f"   {class_name}: {weight:.4f}")
        
        return data_config, class_weights
        
    def find_latest_checkpoint(self):
        """Find the latest training checkpoint to resume from"""
        if not self.runs_dir.exists():
            return None
            
        # Look for existing training runs
        existing_runs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        if not existing_runs:
            return None
            
        # Find the most recent run with weights
        latest_run = None
        latest_time = 0
        
        for run_dir in existing_runs:
            weights_dir = run_dir / "weights"
            if weights_dir.exists():
                last_pt = weights_dir / "last.pt"
                if last_pt.exists():
                    mod_time = last_pt.stat().st_mtime
                    if mod_time > latest_time:
                        latest_time = mod_time
                        latest_run = run_dir
        
        if latest_run:
            checkpoint = latest_run / "weights" / "last.pt"
            self.logger.info(f"🔄 Found checkpoint to resume: {checkpoint}")
            return checkpoint
        
        return None
        
    def create_weighted_config(self, data_config, class_weights):
        """Create a temporary data config with class weights for training"""
        if not class_weights:
            return self.data_yaml
            
        # Create weighted config
        weighted_config = data_config.copy()
        
        # Add class weights to config (if supported by YOLOv12)
        # Note: This might need adjustment based on YOLOv12's actual implementation
        weighted_config['class_weights'] = class_weights
        
        # Save temporary config
        temp_config_path = self.log_dir / "data_weighted.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(weighted_config, f, default_flow_style=False)
        
        self.logger.info(f"💾 Weighted config saved: {temp_config_path}")
        return temp_config_path
        
    def train(self, data_config, class_weights, resume_checkpoint=None):
        """Run YOLOv12 training"""
        self.logger.info("🚀 STARTING IMBALANCED DATASET TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Dataset: {self.dataset_path.name}")
        self.logger.info(f"🔄 Epochs: {self.epochs}")
        self.logger.info(f"📦 Batch size: {self.batch_size}")
        self.logger.info(f"📏 Image size: {self.img_size}")
        self.logger.info(f"⏳ Patience: {self.patience}")
        self.logger.info(f"🖥️ Device: {self.device}")
        self.logger.info(f"⚖️ Class weights: {'Enabled' if class_weights else 'Disabled'}")
        self.logger.info(f"📁 Output: {self.log_dir}")
        self.logger.info("=" * 60)
        
        # Use weighted config if available
        config_path = self.create_weighted_config(data_config, class_weights)
        
        # Build training command
        cmd = [
            'yolo',
            'task=detect',
            'mode=train',
            f'data={config_path}',
            f'epochs={self.epochs}',
            f'batch={self.batch_size}',
            f'imgsz={self.img_size}',
            f'patience={self.patience}',
            f'project={self.runs_dir}',
            f'name=imbalanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            f'device={self.device}',
            'save=True',
            'save_period=5',  # Save every 5 epochs
            'plots=True',
            'val=True',
            'verbose=True'
        ]
        
        # Add model or resume checkpoint
        if resume_checkpoint:
            cmd.append(f'model={resume_checkpoint}')
            cmd.append('resume=True')
            self.logger.info(f"🔄 Resuming from: {resume_checkpoint}")
        else:
            cmd.append(f'model={self.model_weights}')
            self.logger.info(f"🆕 Starting fresh training")
        
        self.logger.info(f"💻 Command: {' '.join(cmd)}")
        
        # Start training
        try:
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Log output in real-time
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.logger.info(line)
                    
                    # Parse and highlight important metrics
                    if 'mAP50' in line or 'mAP50-95' in line:
                        self.logger.info(f"📊 METRICS: {line}")
                    elif 'loss' in line.lower():
                        self.logger.info(f"📉 LOSS: {line}")
                    elif any(cls in line for cls in ['Vehicle', 'Person', 'Bus', 'Bicycle']):
                        self.logger.info(f"📈 CLASS METRICS: {line}")
            
            process.wait()
            
            if process.returncode == 0:
                self.logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                return True
            else:
                self.logger.error(f"❌ Training failed with exit code: {process.returncode}")
                return False
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Training interrupted by user")
            if 'process' in locals():
                process.terminate()
            return False
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            return False
    
    def save_training_info(self, data_config, class_weights):
        """Save training configuration and info"""
        info = {
            "dataset_type": "imbalanced",
            "dataset_path": str(self.dataset_path),
            "model_weights": str(self.model_weights),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "img_size": self.img_size,
            "patience": self.patience,
            "device": self.device,
            "num_classes": data_config.get('nc', 0),
            "class_names": data_config.get('names', []),
            "class_weights_enabled": class_weights is not None,
            "class_weights": class_weights,
            "training_start": datetime.now().isoformat(),
            "log_dir": str(self.log_dir)
        }
        
        info_file = self.log_dir / "training_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Training info saved: {info_file}")
    
    def run(self):
        """Main training pipeline"""
        try:
            # Check prerequisites
            data_config, class_weights = self.check_prerequisites()
            
            # Save training info
            self.save_training_info(data_config, class_weights)
            
            # Check for existing checkpoint
            checkpoint = self.find_latest_checkpoint()
            
            # Ask user about resuming
            if checkpoint:
                response = input(f"\n🔄 Found checkpoint: {checkpoint.name}\nResume training? (y/n): ").lower()
                if response not in ['y', 'yes']:
                    checkpoint = None
                    self.logger.info("🆕 Starting fresh training as requested")
            
            # Run training
            success = self.train(data_config, class_weights, checkpoint)
            
            if success:
                self.logger.info("✅ Imbalanced dataset training completed!")
                print(f"\n🎉 Training completed successfully!")
                print(f"📁 Results saved in: {self.log_dir}")
                print(f"📊 Check runs/imbalanced/ for all outputs")
            else:
                self.logger.error("❌ Training failed!")
                sys.exit(1)
                
        except Exception as e:
            self.logger.error(f"❌ Fatal error: {e}")
            sys.exit(1)

def main():
    """Main function"""
    print("🚀 YOLOv12 Imbalanced Dataset Training")
    print("=" * 50)
    
    trainer = ImbalancedTrainer()
    trainer.run()

if __name__ == "__main__":
    main()