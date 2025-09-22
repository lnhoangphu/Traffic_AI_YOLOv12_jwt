#!/usr/bin/env python3
"""
Compare Results: Balanced vs Imbalanced Training
Script so sánh kết quả training giữa balanced và imbalanced datasets
"""

import json
import csv
import pandas as pd
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class ResultsComparator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.balanced_runs = self.project_root / "runs" / "balanced"
        self.imbalanced_runs = self.project_root / "runs" / "imbalanced"
        self.output_dir = self.project_root / "comparison_results"
        self.output_dir.mkdir(exist_ok=True)
        
    def find_latest_runs(self):
        """Find the latest training runs for both datasets"""
        balanced_run = self.find_latest_run(self.balanced_runs)
        imbalanced_run = self.find_latest_run(self.imbalanced_runs)
        
        return balanced_run, imbalanced_run
    
    def find_latest_run(self, runs_dir):
        """Find the latest training run in a directory"""
        if not runs_dir.exists():
            return None
            
        # Find directories with results
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return None
            
        # Find the most recent one with results.csv
        latest_run = None
        latest_time = 0
        
        for run_dir in run_dirs:
            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                mod_time = results_csv.stat().st_mtime
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_run = run_dir
        
        return latest_run
    
    def load_training_results(self, run_dir):
        """Load training results from a run directory"""
        if not run_dir or not run_dir.exists():
            return None
            
        results = {}
        
        # Load results.csv (main metrics)
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            results['metrics_df'] = df
            
            # Get final metrics (last row)
            if not df.empty:
                final_metrics = df.iloc[-1].to_dict()
                results['final_metrics'] = final_metrics
        
        # Load training info
        info_json = run_dir / "training_info.json"
        if info_json.exists():
            with open(info_json, 'r', encoding='utf-8') as f:
                results['training_info'] = json.load(f)
        
        # Load confusion matrix results if available
        confusion_matrix_png = run_dir / "confusion_matrix.png"
        if confusion_matrix_png.exists():
            results['confusion_matrix_path'] = confusion_matrix_png
            
        # Load per-class results if available
        results_txt = run_dir / "results.txt"
        if results_txt.exists():
            results['results_txt_path'] = results_txt
            
        results['run_dir'] = run_dir
        return results
    
    def extract_class_metrics(self, results):
        """Extract per-class metrics from results"""
        if not results or 'metrics_df' not in results:
            return None
            
        df = results['metrics_df']
        if df.empty:
            return None
            
        # Get the final epoch metrics
        final_row = df.iloc[-1]
        
        class_metrics = {}
        
        # Standard YOLO metrics columns
        for col in df.columns:
            if 'precision' in col.lower() or 'recall' in col.lower() or 'map' in col.lower():
                class_metrics[col] = final_row[col]
        
        return class_metrics
    
    def compare_overall_metrics(self, balanced_results, imbalanced_results):
        """Compare overall metrics between balanced and imbalanced"""
        comparison = {}
        
        if balanced_results and 'final_metrics' in balanced_results:
            balanced_metrics = balanced_results['final_metrics']
        else:
            balanced_metrics = {}
            
        if imbalanced_results and 'final_metrics' in imbalanced_results:
            imbalanced_metrics = imbalanced_results['final_metrics']
        else:
            imbalanced_metrics = {}
        
        # Key metrics to compare
        key_metrics = [
            'mAP_0.5', 'mAP_0.5:0.95', 'precision', 'recall', 
            'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
        ]
        
        for metric in key_metrics:
            balanced_val = balanced_metrics.get(metric, 'N/A')
            imbalanced_val = imbalanced_metrics.get(metric, 'N/A')
            
            comparison[metric] = {
                'balanced': balanced_val,
                'imbalanced': imbalanced_val,
                'difference': 'N/A'
            }
            
            # Calculate difference if both values are numeric
            if isinstance(balanced_val, (int, float)) and isinstance(imbalanced_val, (int, float)):
                diff = imbalanced_val - balanced_val
                diff_pct = (diff / balanced_val * 100) if balanced_val != 0 else 0
                comparison[metric]['difference'] = f"{diff:+.4f} ({diff_pct:+.2f}%)"
        
        return comparison
    
    def generate_comparison_report(self, balanced_results, imbalanced_results):
        """Generate comprehensive comparison report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"training_comparison_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🔬 YOLOV12 TRAINING COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Training info
            if balanced_results and 'training_info' in balanced_results:
                balanced_info = balanced_results['training_info']
                f.write(f"📊 BALANCED DATASET:\n")
                f.write(f"   Dataset: {balanced_info.get('dataset_path', 'N/A')}\n")
                f.write(f"   Training start: {balanced_info.get('training_start', 'N/A')}\n")
                f.write(f"   Epochs: {balanced_info.get('epochs', 'N/A')}\n")
                f.write(f"   Batch size: {balanced_info.get('batch_size', 'N/A')}\n\n")
            
            if imbalanced_results and 'training_info' in imbalanced_results:
                imbalanced_info = imbalanced_results['training_info']
                f.write(f"📊 IMBALANCED DATASET:\n")
                f.write(f"   Dataset: {imbalanced_info.get('dataset_path', 'N/A')}\n")
                f.write(f"   Training start: {imbalanced_info.get('training_start', 'N/A')}\n")
                f.write(f"   Epochs: {imbalanced_info.get('epochs', 'N/A')}\n")
                f.write(f"   Batch size: {imbalanced_info.get('batch_size', 'N/A')}\n")
                f.write(f"   Class weights: {imbalanced_info.get('class_weights_enabled', 'N/A')}\n\n")
            
            # Overall comparison
            comparison = self.compare_overall_metrics(balanced_results, imbalanced_results)
            
            f.write("📈 OVERALL METRICS COMPARISON:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Metric':<20} {'Balanced':<12} {'Imbalanced':<12} {'Difference':<15}\n")
            f.write("-" * 40 + "\n")
            
            for metric, values in comparison.items():
                balanced_str = f"{values['balanced']:.4f}" if isinstance(values['balanced'], (int, float)) else str(values['balanced'])
                imbalanced_str = f"{values['imbalanced']:.4f}" if isinstance(values['imbalanced'], (int, float)) else str(values['imbalanced'])
                
                f.write(f"{metric:<20} {balanced_str:<12} {imbalanced_str:<12} {values['difference']:<15}\n")
            
            f.write("\n")
            
            # Summary and conclusions
            f.write("🎯 SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            # Determine which is better
            if balanced_results and imbalanced_results:
                balanced_map = balanced_results.get('final_metrics', {}).get('mAP_0.5', 0)
                imbalanced_map = imbalanced_results.get('final_metrics', {}).get('mAP_0.5', 0)
                
                if isinstance(balanced_map, (int, float)) and isinstance(imbalanced_map, (int, float)):
                    if balanced_map > imbalanced_map:
                        f.write("🏆 Balanced dataset achieved higher mAP@0.5\n")
                    elif imbalanced_map > balanced_map:
                        f.write("🏆 Imbalanced dataset achieved higher mAP@0.5\n")
                    else:
                        f.write("🤝 Both datasets achieved similar mAP@0.5\n")
            
            f.write(f"\n📁 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"📄 Detailed report saved: {report_file}")
        return report_file
    
    def generate_csv_comparison(self, balanced_results, imbalanced_results):
        """Generate CSV comparison for easy analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.output_dir / f"metrics_comparison_{timestamp}.csv"
        
        comparison = self.compare_overall_metrics(balanced_results, imbalanced_results)
        
        # Prepare data for CSV
        csv_data = []
        for metric, values in comparison.items():
            csv_data.append({
                'Metric': metric,
                'Balanced': values['balanced'],
                'Imbalanced': values['imbalanced'],
                'Difference': values['difference']
            })
        
        # Write CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Metric', 'Balanced', 'Imbalanced', 'Difference'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"📊 CSV comparison saved: {csv_file}")
        return csv_file
    
    def plot_training_curves(self, balanced_results, imbalanced_results):
        """Plot training curves comparison"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves: Balanced vs Imbalanced', fontsize=16)
        
        # Metrics to plot
        metrics = [
            ('train/box_loss', 'Training Box Loss'),
            ('val/box_loss', 'Validation Box Loss'),
            ('mAP_0.5', 'mAP@0.5'),
            ('precision', 'Precision')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Plot balanced
            if balanced_results and 'metrics_df' in balanced_results:
                df_balanced = balanced_results['metrics_df']
                if metric in df_balanced.columns:
                    ax.plot(df_balanced.index, df_balanced[metric], 
                           label='Balanced', color='blue', linewidth=2)
            
            # Plot imbalanced
            if imbalanced_results and 'metrics_df' in imbalanced_results:
                df_imbalanced = imbalanced_results['metrics_df']
                if metric in df_imbalanced.columns:
                    ax.plot(df_imbalanced.index, df_imbalanced[metric], 
                           label='Imbalanced', color='red', linewidth=2)
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / f"training_curves_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves saved: {plot_file}")
        return plot_file
    
    def print_summary_table(self, comparison):
        """Print summary table to terminal"""
        print("\n📊 TRAINING RESULTS COMPARISON")
        print("=" * 80)
        print(f"{'Metric':<25} {'Balanced':<15} {'Imbalanced':<15} {'Difference':<20}")
        print("-" * 80)
        
        for metric, values in comparison.items():
            balanced_str = f"{values['balanced']:.4f}" if isinstance(values['balanced'], (int, float)) else str(values['balanced'])[:12]
            imbalanced_str = f"{values['imbalanced']:.4f}" if isinstance(values['imbalanced'], (int, float)) else str(values['imbalanced'])[:12]
            diff_str = str(values['difference'])[:18]
            
            print(f"{metric:<25} {balanced_str:<15} {imbalanced_str:<15} {diff_str:<20}")
        
        print("=" * 80)
    
    def run_comparison(self):
        """Run complete comparison analysis"""
        print("🔬 Starting Training Results Comparison...")
        print("=" * 50)
        
        # Find latest runs
        balanced_run, imbalanced_run = self.find_latest_runs()
        
        if not balanced_run:
            print("❌ No balanced training results found in runs/balanced/")
            return False
            
        if not imbalanced_run:
            print("❌ No imbalanced training results found in runs/imbalanced/")
            return False
        
        print(f"✅ Found balanced run: {balanced_run.name}")
        print(f"✅ Found imbalanced run: {imbalanced_run.name}")
        
        # Load results
        print("\n📖 Loading training results...")
        balanced_results = self.load_training_results(balanced_run)
        imbalanced_results = self.load_training_results(imbalanced_run)
        
        if not balanced_results:
            print("❌ Failed to load balanced results")
            return False
            
        if not imbalanced_results:
            print("❌ Failed to load imbalanced results")
            return False
        
        print("✅ Results loaded successfully")
        
        # Generate comparison
        print("\n📊 Generating comparison analysis...")
        comparison = self.compare_overall_metrics(balanced_results, imbalanced_results)
        
        # Print to terminal
        self.print_summary_table(comparison)
        
        # Generate outputs
        print(f"\n📁 Saving results to: {self.output_dir}")
        report_file = self.generate_comparison_report(balanced_results, imbalanced_results)
        csv_file = self.generate_csv_comparison(balanced_results, imbalanced_results)
        
        # Try to generate plots
        try:
            plot_file = self.plot_training_curves(balanced_results, imbalanced_results)
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
        
        print(f"\n🎉 Comparison completed successfully!")
        print(f"📂 All results saved in: {self.output_dir}")
        
        return True

def main():
    """Main function"""
    print("🔬 YOLOv12 Training Results Comparison")
    print("=" * 50)
    
    comparator = ResultsComparator()
    success = comparator.run_comparison()
    
    if not success:
        print("\n❌ Comparison failed!")
        print("Make sure you have completed training for both datasets.")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())