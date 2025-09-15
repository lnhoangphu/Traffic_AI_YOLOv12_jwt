"""
Script đánh giá chi tiết model và tạo các biểu đồ visualization.
Phân tích class imbalance và đề xuất cải thiện.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import yaml
import json

plt.style.use('default')
sns.set_palette("husl")

def analyze_dataset_distribution():
    """Phân tích phân phối dataset để hiểu class imbalance"""
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = REPO_ROOT / "data" / "traffic"
    DATA_YAML = DATA_ROOT / "data.yaml"
    
    if not DATA_YAML.exists():
        print("Không tìm thấy data.yaml")
        return
        
    with open(DATA_YAML) as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    
    # Thống kê cho từng split
    stats = {}
    
    for split in ['train', 'val', 'test']:
        label_dir = DATA_ROOT / "labels" / split
        if not label_dir.exists():
            continue
            
        class_counts = defaultdict(int)
        total_instances = 0
        total_files = 0
        empty_files = 0
        
        for label_file in label_dir.glob("*.txt"):
            total_files += 1
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines or not any(line.strip() for line in lines):
                empty_files += 1
            else:
                for line in lines:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
                        total_instances += 1
        
        stats[split] = {
            'class_counts': dict(class_counts),
            'total_instances': total_instances,
            'total_files': total_files,
            'empty_files': empty_files
        }
    
    # Tạo visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Analysis - Traffic Object Detection', fontsize=16, fontweight='bold')
    
    # 1. Class distribution per split
    ax1 = axes[0, 0]
    splits = list(stats.keys())
    class_ids = sorted(set().union(*[s['class_counts'].keys() for s in stats.values()]))
    
    width = 0.25
    x = np.arange(len(class_ids))
    
    for i, split in enumerate(splits):
        counts = [stats[split]['class_counts'].get(cid, 0) for cid in class_ids]
        ax1.bar(x + i*width, counts, width, label=split, alpha=0.8)
    
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Number of Instances')
    ax1.set_title('Class Distribution by Split')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f"{cid}\n{class_names[cid]}" for cid in class_ids], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Total instances per split
    ax2 = axes[0, 1]
    split_totals = [stats[split]['total_instances'] for split in splits]
    colors = plt.cm.Set3(np.linspace(0, 1, len(splits)))
    ax2.pie(split_totals, labels=splits, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Total Instances Distribution')
    
    # 3. Class imbalance ratio
    ax3 = axes[1, 0]
    train_counts = stats.get('train', {}).get('class_counts', {})
    if train_counts:
        total_train = sum(train_counts.values())
        imbalance_ratios = []
        class_labels = []
        
        for class_id in sorted(train_counts.keys()):
            ratio = train_counts[class_id] / total_train * 100
            imbalance_ratios.append(ratio)
            class_labels.append(f"{class_id}: {class_names[class_id]}")
        
        bars = ax3.barh(class_labels, imbalance_ratios, alpha=0.7)
        ax3.set_xlabel('Percentage (%)')
        ax3.set_title('Class Imbalance in Training Set')
        ax3.grid(axis='x', alpha=0.3)
        
        # Color bars based on imbalance severity
        for i, bar in enumerate(bars):
            if imbalance_ratios[i] < 5:  # Very underrepresented
                bar.set_color('red')
            elif imbalance_ratios[i] < 15:  # Underrepresented
                bar.set_color('orange')
            else:  # Well represented
                bar.set_color('green')
    
    # 4. Files vs Instances
    ax4 = axes[1, 1]
    file_counts = [stats[split]['total_files'] for split in splits]
    instance_counts = [stats[split]['total_instances'] for split in splits]
    
    x_pos = np.arange(len(splits))
    width = 0.35
    
    ax4.bar(x_pos - width/2, file_counts, width, label='Files', alpha=0.8)
    ax4.bar(x_pos + width/2, instance_counts, width, label='Instances', alpha=0.8)
    
    ax4.set_xlabel('Dataset Split')
    ax4.set_ylabel('Count')
    ax4.set_title('Files vs Instances')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(splits)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = REPO_ROOT / "analysis"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "dataset_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("=== DATASET ANALYSIS SUMMARY ===")
    for split in splits:
        s = stats[split]
        print(f"\n{split.upper()} SET:")
        print(f"  Total files: {s['total_files']}")
        print(f"  Empty files: {s['empty_files']}")
        print(f"  Total instances: {s['total_instances']}")
        print(f"  Avg instances per file: {s['total_instances']/max(1, s['total_files'] - s['empty_files']):.2f}")
    
    # Class imbalance recommendations
    if 'train' in stats:
        train_counts = stats['train']['class_counts']
        total = sum(train_counts.values())
        
        print(f"\n=== CLASS IMBALANCE ANALYSIS ===")
        underrepresented = []
        for class_id, count in train_counts.items():
            percentage = count / total * 100
            class_name = class_names[class_id]
            print(f"{class_name}: {count} instances ({percentage:.1f}%)")
            
            if percentage < 10:  # Less than 10% is considered underrepresented
                underrepresented.append(class_name)
        
        if underrepresented:
            print(f"\n⚠️  UNDERREPRESENTED CLASSES: {', '.join(underrepresented)}")
            print("💡 RECOMMENDATIONS:")
            print("   1. Collect more data for underrepresented classes")
            print("   2. Apply targeted data augmentation")
            print("   3. Use class weights during training")
            print("   4. Consider synthetic data generation")

def plot_training_results():
    """Vẽ biểu đồ kết quả training"""
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    # Tìm experiment mới nhất
    runs_dir = REPO_ROOT / "runs" / "detect"
    if not runs_dir.exists():
        print("Không tìm thấy kết quả training")
        return
    
    latest_exp = max(runs_dir.glob("traffic_yolov12*"), key=lambda x: x.stat().st_mtime, default=None)
    if not latest_exp:
        print("Không tìm thấy experiment traffic_yolov12")
        return
    
    results_csv = latest_exp / "results.csv"
    if not results_csv.exists():
        print(f"Không tìm thấy {results_csv}")
        return
    
    # Load training results
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    # Create training plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLOv12 Training Results - Traffic Detection', fontsize=16, fontweight='bold')
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        ax1.plot(df.index, df['train/box_loss'], label='Train Box Loss', linewidth=2)
        ax1.plot(df.index, df['val/box_loss'], label='Val Box Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Box Loss')
        ax1.set_title('Box Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # 2. Objectness loss
    ax2 = axes[0, 1]
    if 'train/obj_loss' in df.columns and 'val/obj_loss' in df.columns:
        ax2.plot(df.index, df['train/obj_loss'], label='Train Obj Loss', linewidth=2)
        ax2.plot(df.index, df['val/obj_loss'], label='Val Obj Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Objectness Loss')
        ax2.set_title('Objectness Loss')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # 3. Class loss
    ax3 = axes[0, 2]
    if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
        ax3.plot(df.index, df['train/cls_loss'], label='Train Cls Loss', linewidth=2)
        ax3.plot(df.index, df['val/cls_loss'], label='Val Cls Loss', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Classification Loss')
        ax3.set_title('Classification Loss')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # 4. mAP curves
    ax4 = axes[1, 0]
    if 'metrics/mAP50(B)' in df.columns:
        ax4.plot(df.index, df['metrics/mAP50(B)'], label='mAP50', linewidth=2, color='green')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax4.plot(df.index, df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2, color='blue')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('mAP')
        ax4.set_title('Mean Average Precision')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    # 5. Precision & Recall
    ax5 = axes[1, 1]
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        ax5.plot(df.index, df['metrics/precision(B)'], label='Precision', linewidth=2, color='orange')
        ax5.plot(df.index, df['metrics/recall(B)'], label='Recall', linewidth=2, color='red')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Score')
        ax5.set_title('Precision & Recall')
        ax5.legend()
        ax5.grid(alpha=0.3)
    
    # 6. Learning Rate
    ax6 = axes[1, 2]
    if 'lr/pg0' in df.columns:
        ax6.plot(df.index, df['lr/pg0'], label='Learning Rate', linewidth=2, color='purple')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Learning Rate')
        ax6.set_title('Learning Rate Schedule')
        ax6.legend()
        ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = REPO_ROOT / "analysis"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "training_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print best metrics
    print("=== BEST TRAINING METRICS ===")
    if 'metrics/mAP50(B)' in df.columns:
        best_map50 = df['metrics/mAP50(B)'].max()
        best_epoch = df['metrics/mAP50(B)'].idxmax()
        print(f"Best mAP50: {best_map50:.3f} at epoch {best_epoch}")
    
    if 'metrics/mAP50-95(B)' in df.columns:
        best_map50_95 = df['metrics/mAP50-95(B)'].max()
        best_epoch_95 = df['metrics/mAP50-95(B)'].idxmax()
        print(f"Best mAP50-95: {best_map50_95:.3f} at epoch {best_epoch_95}")

def main():
    """Main analysis function"""
    
    print("=== TRAFFIC AI MODEL ANALYSIS ===")
    
    # 1. Analyze dataset
    print("\n1. Analyzing dataset distribution...")
    analyze_dataset_distribution()
    
    # 2. Plot training results
    print("\n2. Plotting training results...")
    plot_training_results()
    
    print("\n=== ANALYSIS COMPLETED ===")
    print("Kiểm tra thư mục analysis/ để xem các biểu đồ đã tạo")

if __name__ == "__main__":
    main()