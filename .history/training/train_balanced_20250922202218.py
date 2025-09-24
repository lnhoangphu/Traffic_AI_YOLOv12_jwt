#!/usr/bin/env python3
"""
YOLOv12 Training Script for Balanced Dataset
Script huấn luyện YOLOv12 cho dataset balanced với checkpoint resume
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

# Fix encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class BalancedTrainer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.dataset_path = self.project_root / "datasets" / "traffic_ai_balanced_11class_processed"
        self.data_yaml = self.dataset_path / "data.yaml"
        self.runs_dir = self.project_root / "runs" / "balanced"
        self.model_weights = self.project_root / "yolo12n.pt"
        
        # Training parameters (optimized for RTX 3050 Ti 4GB)
        self.epochs = 100
        self.batch_size = 4  # Reduced from 16 to fit in 4GB VRAM
        self.img_size = 640
        self.patience = 20
        self.device = "0"
        
        # Logging setup
        self.setup_logging()
        self.setup_signal_handlers()
        
    def setup_logging(self):
        """Setup logging to file and console"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.runs_dir / f"balanced_training_{timestamp}"
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
        
        return data_config
        
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
        
    def train(self, resume_checkpoint=None, finetune=False):
        """Run YOLOv12 training"""
        self.logger.info("🚀 STARTING BALANCED DATASET TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Dataset: {self.dataset_path.name}")
        self.logger.info(f"🔄 Epochs: {self.epochs}")
        self.logger.info(f"📦 Batch size: {self.batch_size}")
        self.logger.info(f"📏 Image size: {self.img_size}")
        self.logger.info(f"⏳ Patience: {self.patience}")
        self.logger.info(f"🖥️ Device: {self.device}")
        self.logger.info(f"📁 Output: {self.log_dir}")
        self.logger.info("=" * 60)
        
        # Build training command
        cmd = [
            'yolo',
            'task=detect',
            'mode=train',
            f'data={self.data_yaml}',
            f'epochs={self.epochs}',
            f'batch={self.batch_size}',
            f'imgsz={self.img_size}',
            f'patience={self.patience}',
            f'project={self.runs_dir}',
            f'name=balanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            f'device={self.device}',
            'save=True',
            'save_period=10',  # Save every 10 epochs instead of 5 to save disk space
            'plots=True',
            'val=True',
            'verbose=False',  # Reduce console output
            'amp=True',  # Enable Automatic Mixed Precision to save VRAM
            'cache=False',  # Disable cache to save RAM
            'workers=4'  # Reduce workers to save RAM
        ]
        
        # Add model or resume/finetune checkpoint
        if resume_checkpoint and not finetune:
            cmd.append(f'model={resume_checkpoint}')
            cmd.append('resume=True')
            self.logger.info(f"🔄 Resuming from: {resume_checkpoint}")
        elif resume_checkpoint and finetune:
            # Start new training initialized from checkpoint weights (no resume)
            cmd.append(f'model={resume_checkpoint}')
            self.logger.info(f"✨ Fine-tuning from: {resume_checkpoint}")
        else:
            cmd.append(f'model={self.model_weights}')
            self.logger.info(f"🆕 Starting fresh training")
        
        self.logger.info(f"💻 Command: {' '.join(cmd)}")
        
        # Set environment variables for proper encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Start training
        try:
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            
            # Log output in real-time
            for line in process.stdout:
                try:
                    line = line.strip()
                    if line:
                        self.logger.info(line)
                        
                        # Parse and highlight important metrics
                        if 'mAP50' in line or 'mAP50-95' in line:
                            self.logger.info(f"📊 METRICS: {line}")
                        elif 'loss' in line.lower():
                            self.logger.info(f"📉 LOSS: {line}")
                except UnicodeDecodeError as e:
                    self.logger.warning(f"⚠️ Encoding error in output: {e}")
                    continue
            
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
    
    def save_training_info(self, data_config):
        """Save training configuration and info"""
        info = {
            "dataset_type": "balanced",
            "dataset_path": str(self.dataset_path),
            "model_weights": str(self.model_weights),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "img_size": self.img_size,
            "patience": self.patience,
            "device": self.device,
            "num_classes": data_config.get('nc', 0),
            "class_names": data_config.get('names', []),
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
            data_config = self.check_prerequisites()
            
            # Save training info
            self.save_training_info(data_config)
            
            # Check for existing checkpoint
            checkpoint = self.find_latest_checkpoint()

            finetune = False
            # Ask user about action
            if checkpoint:
                ft_epochs = int(os.getenv('FT_EPOCHS', '20'))
                print(f"\n🔎 Found checkpoint: {checkpoint.name}")
                print("Choose: [r]esume (continue same run), [f]inetune (start new run from weights), [n]ew (fresh)")
                response = input(f"Your choice (r/f/n, default=f, epochs={ft_epochs}): ").strip().lower() or 'f'
                if response.startswith('r'):
                    finetune = False
                elif response.startswith('f'):
                    finetune = True
                    # Override epochs for quicker fix if FT_EPOCHS set
                    try:
                        self.epochs = int(os.getenv('FT_EPOCHS', str(self.epochs)))
                    except Exception:
                        pass
                else:
                    checkpoint = None
                    self.logger.info("🆕 Starting fresh training as requested")

            # Run training
            success = self.train(checkpoint, finetune=finetune)
            
            if success:
                self.logger.info("✅ Balanced dataset training completed!")
                print(f"\n🎉 Training completed successfully!")
                print(f"📁 Results saved in: {self.log_dir}")
                print(f"📊 Check runs/balanced/ for all outputs")
            else:
                self.logger.error("❌ Training failed!")
                sys.exit(1)
                
        except Exception as e:
            self.logger.error(f"❌ Fatal error: {e}")
            sys.exit(1)

def main():
    """Main function"""
    print("🚀 YOLOv12 Balanced Dataset Training")
    print("=" * 50)
    
    trainer = BalancedTrainer()
    trainer.run()

if __name__ == "__main__":
    main()