"""
Phân tích chất lượng labels trong dataset
Tìm các trường hợp có thể bị label nhầm giữa Vehicle và Person
"""

import os
from pathlib import Path
from collections import defaultdict
import json

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "traffic_ai_final_balanced"

def analyze_labels():
    """Phân tích distribution và tìm patterns bất thường"""
    
    stats = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int)
    }
    
    suspicious_files = []
    
    class_names = {
        0: "Vehicle", 1: "Bus", 2: "Bicycle", 3: "Person", 
        4: "Engine", 5: "Truck", 6: "Tricycle", 7: "Obstacle",
        8: "Pothole", 9: "Traffic Light", 10: "Traffic Sign"
    }
    
    for split in ["train", "val", "test"]:
        labels_dir = DATASET_PATH / "labels" / split
        
        if not labels_dir.exists():
            print(f"⚠️  {split} labels không tồn tại!")
            continue
        
        label_files = list(labels_dir.glob("*.txt"))
        print(f"\n📊 Analyzing {split} split: {len(label_files)} files")
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            file_stats = defaultdict(int)
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    stats[split][class_id] += 1
                    file_stats[class_id] += 1
            
            # Detect suspicious patterns
            person_count = file_stats[3]  # Person
            vehicle_count = file_stats[0]  # Vehicle
            
            # Pattern 1: Quá nhiều Person, quá ít Vehicle (có thể là traffic scene)
            if person_count > 10 and vehicle_count < 3:
                suspicious_files.append({
                    "file": str(label_file.name),
                    "split": split,
                    "reason": "Too many Person, too few Vehicle in traffic scene",
                    "person_count": person_count,
                    "vehicle_count": vehicle_count
                })
            
            # Pattern 2: Chỉ có Person, không có vehicle nào (rất nghi ngờ)
            if person_count > 5 and vehicle_count == 0 and file_stats[4] == 0:  # No Engine too
                suspicious_files.append({
                    "file": str(label_file.name),
                    "split": split,
                    "reason": "Only Person detected in traffic image (likely mislabeled vehicles)",
                    "person_count": person_count,
                    "vehicle_count": 0
                })
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 DATASET LABEL STATISTICS")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}:")
        total = sum(stats[split].values())
        for class_id in sorted(stats[split].keys()):
            count = stats[split][class_id]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {class_names.get(class_id, f'Class {class_id}'):15s}: {count:7d} ({percentage:5.2f}%)")
    
    # Print suspicious files
    print("\n" + "="*60)
    print(f"🚨 SUSPICIOUS FILES: {len(suspicious_files)}")
    print("="*60)
    
    if suspicious_files:
        # Group by reason
        by_reason = defaultdict(list)
        for item in suspicious_files:
            by_reason[item["reason"]].append(item)
        
        for reason, items in by_reason.items():
            print(f"\n❌ {reason}: {len(items)} files")
            for item in items[:5]:  # Show first 5 examples
                print(f"   - {item['file']} ({item['split']}): "
                      f"Person={item['person_count']}, Vehicle={item['vehicle_count']}")
            if len(items) > 5:
                print(f"   ... and {len(items)-5} more files")
    
    # Save report
    report = {
        "statistics": {split: dict(stats[split]) for split in ["train", "val", "test"]},
        "suspicious_files": suspicious_files,
        "total_suspicious": len(suspicious_files)
    }
    
    report_path = PROJECT_ROOT / "label_quality_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Report saved to: {report_path}")
    
    # Recommendations
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    person_ratio = stats["train"][3] / sum(stats["train"].values()) * 100 if sum(stats["train"].values()) > 0 else 0
    vehicle_ratio = stats["train"][0] / sum(stats["train"].values()) * 100 if sum(stats["train"].values()) > 0 else 0
    
    if person_ratio > 30 and vehicle_ratio < 15:
        print("⚠️  Person ratio quá cao, Vehicle quá thấp")
        print("   → Dataset có thể bị mislabel xe thành người")
        print("   → Cần review và fix labels")
    
    if len(suspicious_files) > 100:
        print(f"⚠️  Có {len(suspicious_files)} files nghi ngờ mislabel")
        print("   → Chạy scripts/fix_vehicle_person_labels.py để sửa")
        print("   → Hoặc retrain model trên dataset sạch hơn")
    
    return report

if __name__ == "__main__":
    print("🔍 ANALYZING LABEL QUALITY...")
    report = analyze_labels()
    print("\n✅ Analysis complete!")
