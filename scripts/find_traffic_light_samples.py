"""
Find Traffic Light Samples in Object Detection 35 Dataset
Tìm tất cả images và labels có class Traffic Light (class 20)

Usage: python scripts/find_traffic_light_samples.py
"""

import os
from pathlib import Path
from collections import defaultdict
import shutil

class TrafficLightFinder:
    def __init__(self):
        self.dataset_path = Path("datasets_src/object_detection_35_organized")
        self.traffic_light_class_id = 20  # Traffic Light class in OD35
        
        # Results
        self.traffic_light_files = []
        self.stats = defaultdict(int)
        
    def find_traffic_light_samples(self):
        """Tìm tất cả files có Traffic Light annotations"""
        print("🔍 SEARCHING FOR TRAFFIC LIGHT SAMPLES")
        print("=" * 50)
        
        # Search in all splits
        for split in ['train', 'val', 'test']:
            label_dir = self.dataset_path / 'labels' / split
            image_dir = self.dataset_path / 'images' / split
            
            if not label_dir.exists():
                print(f"⚠️ Label directory not found: {label_dir}")
                continue
                
            print(f"\n📂 Searching in {split} split...")
            
            # Go through all label files
            for label_file in label_dir.glob('*.txt'):
                has_traffic_light = False
                traffic_light_count = 0
                
                # Read label file
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id == self.traffic_light_class_id:
                                has_traffic_light = True
                                traffic_light_count += 1
                                
                except Exception as e:
                    print(f"   ⚠️ Error reading {label_file}: {e}")
                    continue
                
                # If has traffic light, save info
                if has_traffic_light:
                    image_file = image_dir / f"{label_file.stem}.jpg"
                    
                    self.traffic_light_files.append({
                        'split': split,
                        'image_path': image_file,
                        'label_path': label_file,
                        'traffic_light_count': traffic_light_count,
                        'total_annotations': len(lines)
                    })
                    
                    self.stats[split] += 1
                    self.stats['total_traffic_lights'] += traffic_light_count
            
            print(f"   Found {self.stats[split]} files with Traffic Light in {split}")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Total files with Traffic Light: {len(self.traffic_light_files)}")
        print(f"   Total Traffic Light annotations: {self.stats['total_traffic_lights']}")
        
        for split in ['train', 'val', 'test']:
            if self.stats[split] > 0:
                print(f"   {split}: {self.stats[split]} files")
    
    def show_sample_details(self, max_samples=10):
        """Show detailed info of first few samples"""
        print(f"\n📋 SAMPLE DETAILS (first {max_samples}):")
        print("-" * 80)
        
        for i, sample in enumerate(self.traffic_light_files[:max_samples]):
            print(f"\n{i+1}. {sample['split'].upper()} - {sample['image_path'].name}")
            print(f"   Image: {sample['image_path']}")
            print(f"   Label: {sample['label_path']}")
            print(f"   Traffic Lights: {sample['traffic_light_count']}")
            print(f"   Total annotations: {sample['total_annotations']}")
            
            # Show label content
            try:
                with open(sample['label_path'], 'r') as f:
                    lines = f.readlines()
                print(f"   Label content:")
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id == self.traffic_light_class_id:
                            print(f"      ➤ Traffic Light: {line.strip()}")
                        else:
                            # Get class name from data.yaml
                            print(f"      - Class {class_id}: {line.strip()}")
            except Exception as e:
                print(f"      ⚠️ Error reading label: {e}")
    
    def copy_traffic_light_samples(self, output_dir="traffic_light_samples", max_copies=20):
        """Copy some Traffic Light samples to separate folder for inspection"""
        output_path = Path(output_dir)
        
        # Create output directory
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📋 COPYING {min(max_copies, len(self.traffic_light_files))} TRAFFIC LIGHT SAMPLES")
        print(f"Output directory: {output_path.absolute()}")
        
        # Copy samples
        for i, sample in enumerate(self.traffic_light_files[:max_copies]):
            try:
                # Copy image
                output_img = output_path / f"tl_{i+1:03d}_{sample['split']}_{sample['image_path'].name}"
                shutil.copy2(sample['image_path'], output_img)
                
                # Copy label
                output_label = output_path / f"tl_{i+1:03d}_{sample['split']}_{sample['label_path'].name}"
                shutil.copy2(sample['label_path'], output_label)
                
                print(f"   ✅ Copied: {sample['image_path'].name} ({sample['traffic_light_count']} traffic lights)")
                
            except Exception as e:
                print(f"   ⚠️ Error copying {sample['image_path'].name}: {e}")
        
        print(f"\n✅ Completed! Check samples in: {output_path.absolute()}")
    
    def create_traffic_light_report(self):
        """Create detailed report"""
        report = f"""
🚦 TRAFFIC LIGHT ANALYSIS REPORT
{'=' * 60}

📊 DATASET: Object Detection 35 (VisionGuard)
   Traffic Light Class ID: {self.traffic_light_class_id}
   Class Name: Traffic Light

📈 STATISTICS:
   Total files with Traffic Light: {len(self.traffic_light_files)}
   Total Traffic Light annotations: {self.stats['total_traffic_lights']}
   
   Distribution by split:
"""
        
        for split in ['train', 'val', 'test']:
            if self.stats[split] > 0:
                percentage = (self.stats[split] / len(self.traffic_light_files)) * 100
                report += f"   - {split}: {self.stats[split]} files ({percentage:.1f}%)\n"
        
        report += f"\n📋 ALL TRAFFIC LIGHT FILES:\n"
        report += f"{'#':<4} {'Split':<6} {'File Name':<30} {'TL Count':<8} {'Total Ann':<10}\n"
        report += "-" * 70 + "\n"
        
        for i, sample in enumerate(self.traffic_light_files):
            report += f"{i+1:<4} {sample['split']:<6} {sample['image_path'].name:<30} {sample['traffic_light_count']:<8} {sample['total_annotations']:<10}\n"
        
        # Save report
        report_path = Path("traffic_light_analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n💾 Report saved to: {report_path.absolute()}")
    
    def run_analysis(self):
        """Run complete analysis"""
        self.find_traffic_light_samples()
        
        if self.traffic_light_files:
            self.show_sample_details()
            self.copy_traffic_light_samples()
            self.create_traffic_light_report()
        else:
            print("\n❌ No Traffic Light samples found!")

if __name__ == "__main__":
    finder = TrafficLightFinder()
    finder.run_analysis()