#!/usr/bin/env python3
"""
YOLOv12 Training Script for Traffic AI Dataset
Script training hoàn chỉnh cho dữ liệu đã tiền xử lý
"""

import os
import yaml
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime

def load_yaml(file_path):
    """Load YAML file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data, file_path):
    """Save YAML file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)

def setup_training_config(data_yaml_path, class_weights_path=None):
    """Setup training configuration"""
    print(f"🔧 Setting up training configuration...")
    
    # Load data config
    data_config = load_yaml(data_yaml_path)
    print(f"📊 Dataset: {data_config.get('path', 'Unknown')}")
    print(f"📊 Classes: {data_config.get('nc', 0)}")
    
    # Verify paths exist
    dataset_path = Path(data_config['path'])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    train_path = dataset_path / data_config['train']
    val_path = dataset_path / data_config['val']
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train path not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val path not found: {val_path}")
    
    print(f"✅ Train images: {len(list(train_path.glob('*.jpg')))}")
    print(f"✅ Val images: {len(list(val_path.glob('*.jpg')))}")
    
    # Load class weights if provided
    class_weights = None
    if class_weights_path and Path(class_weights_path).exists():
        weights_data = load_yaml(class_weights_path)
        class_weights = weights_data.get('class_weights', {})
        print(f"⚖️ Using class weights from: {class_weights_path}")
    
    return data_config, class_weights

def run_training(
    data_yaml_path,
    model='yolov12n.pt',
    epochs=100,
    batch_size=16,
    img_size=640,
    patience=20,
    class_weights_path=None,
    project='runs/train',
    name=None,
    resume=False,
    device='0'
):
    """Run YOLOv12 training"""
    
    # Setup configuration
    data_config, class_weights = setup_training_config(data_yaml_path, class_weights_path)
    
    # Generate run name if not provided
    if name is None:
        dataset_name = Path(data_config['path']).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{dataset_name}_{timestamp}"
    
    print(f"\n🚀 STARTING TRAINING")
    print("=" * 60)
    print(f"📊 Dataset: {data_yaml_path}")
    print(f"🤖 Model: {model}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📏 Image size: {img_size}")
    print(f"⏳ Patience: {patience}")
    print(f"🖥️ Device: {device}")
    print(f"📁 Project: {project}")
    print(f"🏷️ Name: {name}")
    if class_weights:
        print(f"⚖️ Class weights: Enabled")
    print("=" * 60)
    
    # Build training command
    cmd = [
        'yolo',
        'task=detect',
        'mode=train',
        f'model={model}',
        f'data={data_yaml_path}',
        f'epochs={epochs}',
        f'batch={batch_size}',
        f'imgsz={img_size}',
        f'patience={patience}',
        f'project={project}',
        f'name={name}',
        f'device={device}',
        'save=True',
        'save_period=10',
        'plots=True',
        'val=True'
    ]
    
    # Add resume if specified
    if resume:
        cmd.append('resume=True')
    
    # Execute training
    try:
        print(f"💻 Command: {' '.join(cmd)}")
        print(f"🏁 Starting training...")
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved in: {project}/{name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TRAINING FAILED!")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv12 Training for Traffic AI')
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml file')
    
    # Optional arguments
    parser.add_argument('--model', type=str, default='yolov12n.pt',
                       help='Model weights path (default: yolov12n.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--class-weights', type=str,
                       help='Path to class weights YAML file')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory (default: runs/train)')
    parser.add_argument('--name', type=str,
                       help='Run name (auto-generated if not specified)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (default: 0)')
    
    args = parser.parse_args()
    
    # Validate data file
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)
    
    # Validate model file
    if not Path(args.model).exists() and not args.resume:
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Run training
    success = run_training(
        data_yaml_path=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        patience=args.patience,
        class_weights_path=args.class_weights,
        project=args.project,
        name=args.name,
        resume=args.resume,
        device=args.device
    )
    
    if success:
        print(f"\n✅ Training completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()