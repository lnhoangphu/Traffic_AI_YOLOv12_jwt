"""
Analysis Script for Balanced vs Imbalanced Results
Phân tích và so sánh kết quả training giữa balanced và imbalanced datasets

Usage: python scripts/analyze_balanced_vs_imbalanced.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import yaml
import json

class ResultsAnalyzer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.balanced_results = self.project_root / "training/runs/balanced/yolov12_balanced_experiment"
        self.imbalanced_results = self.project_root / "training/runs/imbalanced/yolov12_imbalanced_experiment"
        self.analysis_output = self.project_root / "training/analysis"
        
        # Create analysis output directory
        self.analysis_output.mkdir(parents=True, exist_ok=True)
        
        # Load class names
        self.load_class_names()

    def load_class_names(self):
        """Load class names from dataset configuration"""
        try:
            balanced_yaml = self.project_root / "datasets/traffic_ai_balanced_11class/data.yaml"
            with open(balanced_yaml, 'r') as f:
                config = yaml.safe_load(f)
            self.class_names = config['names']
            self.num_classes = len(self.class_names)
            print(f"✅ Loaded {self.num_classes} class names")
        except Exception as e:
            print(f"⚠️ Could not load class names: {e}")
            self.class_names = [f"Class_{i}" for i in range(11)]
            self.num_classes = 11

    def load_training_results(self):
        """Load training results from both experiments"""
        results = {}
        
        # Load balanced results
        balanced_csv = self.balanced_results / "results.csv"
        if balanced_csv.exists():
            results['balanced'] = pd.read_csv(balanced_csv)
            print(f"✅ Loaded balanced results: {len(results['balanced'])} epochs")
        else:
            print(f"❌ Balanced results not found: {balanced_csv}")
            
        # Load imbalanced results
        imbalanced_csv = self.imbalanced_results / "results.csv"
        if imbalanced_csv.exists():
            results['imbalanced'] = pd.read_csv(imbalanced_csv)
            print(f"✅ Loaded imbalanced results: {len(results['imbalanced'])} epochs")
        else:
            print(f"❌ Imbalanced results not found: {imbalanced_csv}")
            
        return results

    def plot_training_curves(self, results):
        """Plot training curves comparison"""
        print("📈 Creating training curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Balanced vs Imbalanced Training Comparison', fontsize=16)
        
        # Define metrics to plot
        metrics = {
            'mAP@0.5': ('metrics/mAP50(B)', 'mAP@0.5'),
            'mAP@0.5:0.95': ('metrics/mAP50-95(B)', 'mAP@0.5:0.95'),
            'Train Loss': ('train/box_loss', 'Box Loss'),
            'Val Loss': ('val/box_loss', 'Validation Box Loss')
        }
        
        for idx, (title, (col_name, ylabel)) in enumerate(metrics.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Plot both datasets
            for dataset_type, df in results.items():
                if col_name in df.columns:
                    epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
                    ax.plot(epochs, df[col_name], label=f'{dataset_type.title()}', linewidth=2)
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_output / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training curves saved: {self.analysis_output / 'training_curves.png'}")

    def analyze_final_metrics(self, results):
        """Analyze final epoch metrics"""
        print("📊 Analyzing final metrics...")
        
        final_metrics = {}
        
        for dataset_type, df in results.items():
            if len(df) == 0:
                continue
                
            final_epoch = df.iloc[-1]
            
            final_metrics[dataset_type] = {
                'mAP50': final_epoch.get('metrics/mAP50(B)', 0),
                'mAP50_95': final_epoch.get('metrics/mAP50-95(B)', 0),
                'precision': final_epoch.get('metrics/precision(B)', 0),
                'recall': final_epoch.get('metrics/recall(B)', 0),
                'final_epoch': final_epoch.get('epoch', 0),
                'train_box_loss': final_epoch.get('train/box_loss', 0),
                'val_box_loss': final_epoch.get('val/box_loss', 0)
            }
        
        # Create comparison table
        comparison_df = pd.DataFrame(final_metrics).T
        
        # Save to CSV
        comparison_df.to_csv(self.analysis_output / 'final_metrics_comparison.csv')
        
        print("📋 Final Metrics Comparison:")
        print(comparison_df.round(4))
        
        return final_metrics

    def plot_metrics_comparison(self, final_metrics):
        """Create bar chart comparing final metrics"""
        print("📊 Creating metrics comparison chart...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Final Metrics: Balanced vs Imbalanced', fontsize=16)
        
        metrics_to_plot = ['mAP50', 'mAP50_95', 'precision', 'recall']
        metric_labels = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            values = [final_metrics.get('balanced', {}).get(metric, 0),
                     final_metrics.get('imbalanced', {}).get(metric, 0)]
            
            bars = ax.bar(['Balanced', 'Imbalanced'], values, 
                         color=['skyblue', 'lightcoral'], alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(label)
            ax.set_ylabel('Score')
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_output / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Metrics comparison saved: {self.analysis_output / 'metrics_comparison.png'}")

    def analyze_per_class_performance(self):
        """Analyze per-class performance if available"""
        print("🔍 Analyzing per-class performance...")
        
        # Look for confusion matrices or per-class results
        per_class_files = {
            'balanced': self.balanced_results / "confusion_matrix.png",
            'imbalanced': self.imbalanced_results / "confusion_matrix.png"
        }
        
        available_files = {}
        for dataset_type, file_path in per_class_files.items():
            if file_path.exists():
                available_files[dataset_type] = file_path
                print(f"✅ Found confusion matrix: {file_path}")
            else:
                print(f"⚠️ Confusion matrix not found: {file_path}")
        
        return available_files

    def create_comprehensive_report(self, results, final_metrics):
        """Create comprehensive analysis report"""
        print("📝 Creating comprehensive report...")
        
        report = f"""
🎯 BALANCED vs IMBALANCED DATASET ANALYSIS REPORT
{'=' * 70}

📊 EXPERIMENT OVERVIEW:
   Purpose: Compare YOLOv12 performance on balanced vs imbalanced traffic datasets
   Classes: {self.num_classes} traffic object classes
   Model: YOLOv12 Nano

📈 FINAL PERFORMANCE METRICS:
"""
        
        if 'balanced' in final_metrics and 'imbalanced' in final_metrics:
            balanced = final_metrics['balanced']
            imbalanced = final_metrics['imbalanced']
            
            report += f"""
   BALANCED DATASET:
      mAP@0.5: {balanced.get('mAP50', 0):.4f}
      mAP@0.5:0.95: {balanced.get('mAP50_95', 0):.4f}
      Precision: {balanced.get('precision', 0):.4f}
      Recall: {balanced.get('recall', 0):.4f}
      Training Epochs: {balanced.get('final_epoch', 0)}

   IMBALANCED DATASET:
      mAP@0.5: {imbalanced.get('mAP50', 0):.4f}
      mAP@0.5:0.95: {imbalanced.get('mAP50_95', 0):.4f}
      Precision: {imbalanced.get('precision', 0):.4f}
      Recall: {imbalanced.get('recall', 0):.4f}
      Training Epochs: {imbalanced.get('final_epoch', 0)}

📊 PERFORMANCE COMPARISON:
   mAP@0.5 Difference: {balanced.get('mAP50', 0) - imbalanced.get('mAP50', 0):+.4f}
   mAP@0.5:0.95 Difference: {balanced.get('mAP50_95', 0) - imbalanced.get('mAP50_95', 0):+.4f}
   Precision Difference: {balanced.get('precision', 0) - imbalanced.get('precision', 0):+.4f}
   Recall Difference: {balanced.get('recall', 0) - imbalanced.get('recall', 0):+.4f}
"""
        
        report += f"""
🔍 KEY FINDINGS:
   1. Data Balance Impact: {'Balanced dataset shows superior performance' if final_metrics.get('balanced', {}).get('mAP50', 0) > final_metrics.get('imbalanced', {}).get('mAP50', 0) else 'Imbalanced dataset performs better or similarly'}
   
   2. Training Stability: Check training curves for convergence patterns
   
   3. Class-specific Impact: Examine confusion matrices for minority class performance
   
   4. Practical Implications: Consider real-world deployment scenarios

📋 RESEARCH CONCLUSIONS:
   • Balancing Effect: {'Positive' if final_metrics.get('balanced', {}).get('mAP50', 0) > final_metrics.get('imbalanced', {}).get('mAP50', 0) else 'Neutral/Negative'} impact on overall performance
   
   • Recommendation: {'Use balanced dataset for better minority class detection' if final_metrics.get('balanced', {}).get('mAP50', 0) > final_metrics.get('imbalanced', {}).get('mAP50', 0) else 'Natural imbalance may be sufficient'}
   
   • Future Work: Investigate advanced balancing techniques (focal loss, class weights)

📂 GENERATED FILES:
   - training_curves.png: Training progress comparison
   - metrics_comparison.png: Final metrics visualization
   - final_metrics_comparison.csv: Detailed metrics table
   - analysis_report.txt: This comprehensive report

🎯 NEXT STEPS:
   1. Validate findings with additional test data
   2. Analyze per-class performance in detail
   3. Consider hybrid approaches (weighted loss, data augmentation)
   4. Test model performance on real traffic scenarios
"""
        
        # Save report
        report_file = self.analysis_output / "analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"✅ Comprehensive report saved: {report_file}")

    def run_analysis(self):
        """Run complete analysis"""
        print("🔬 BALANCED vs IMBALANCED RESULTS ANALYSIS")
        print("=" * 60)
        
        # Load training results
        results = self.load_training_results()
        
        if not results:
            print("❌ No training results found to analyze!")
            return
        
        # Plot training curves
        if results:
            self.plot_training_curves(results)
        
        # Analyze final metrics
        final_metrics = self.analyze_final_metrics(results)
        
        # Create metrics comparison
        if final_metrics:
            self.plot_metrics_comparison(final_metrics)
        
        # Analyze per-class performance
        self.analyze_per_class_performance()
        
        # Create comprehensive report
        self.create_comprehensive_report(results, final_metrics)
        
        print(f"\\n🎉 ANALYSIS COMPLETED!")
        print(f"📁 Results saved in: {self.analysis_output}")

if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    analyzer.run_analysis()