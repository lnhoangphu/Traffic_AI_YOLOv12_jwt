"""
Analyze Object Detection 35 Classes Dataset - Correct Version
Phân tích dataset Object Detection 35 để xác định mapping từ class IDs sang class names
theo mô tả chính thức: Person, Chair, Toothbrush, Knife, Bottle, Cup, Spoon, etc.

Usage: python scripts/analyze_object_detection_35_correct.py
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import random

class ObjectDetection35Analyzer:
    def __init__(self):
        self.dataset_root = Path("datasets_src/object_detection_35/final batches")
        
        # Class names theo mô tả chính thức từ Kaggle
        self.official_class_names = [
            "Person",           # 0
            "Chair",            # 1  
            "Toothbrush",       # 2
            "Knife",            # 3
            "Bottle",           # 4
            "Cup",              # 5
            "Spoon",            # 6
            "Bench",            # 7
            "Refrigerator",     # 8
            "Fork",             # 9
            "Bus",              # 10
            "Toilet",           # 11
            "Bicycle",          # 12
            "Airplane",         # 13
            "Truck",            # 14
            "Motorcycles",      # 15 (Note: plural form)
            "Oven",             # 16
            "Dog",              # 17
            "Bed",              # 18
            "Cat",              # 19
            "Traffic Light",    # 20
            "Currency",         # 21
            "Face",             # 22
            "Stop Sign",        # 23
            "Car",              # 24
            "Barriers",         # 25
            "Path Holes",       # 26
            "Stairs",           # 27
            "Train",            # 28
            "Bin",              # 29
            "Blind Stick",      # 30
            "Men Sign",         # 31
            "Cell Phone",       # 32
            "Women Sign",       # 33
            "Tap"               # 34
        ]
        
        # Traffic-related classes for our taxonomy
        self.traffic_related_classes = {
            0: "Person",         # pedestrian
            10: "Bus",           # bus  
            12: "Bicycle",       # bicycle
            14: "Truck",         # truck
            15: "Motorcycles",   # motorcycle
            20: "Traffic Light", # traffic_light
            23: "Stop Sign",     # traffic_sign
            24: "Car",           # car
            25: "Barriers",      # traffic_infrastructure
            26: "Path Holes",    # pothole
            28: "Train",         # train
        }
    
    def analyze_class_distribution(self):
        """Phân tích phân phối class trong dataset"""
        print("🔍 ANALYZING OBJECT DETECTION 35 CLASSES")
        print("=" * 60)
        
        class_counts = Counter()
        total_annotations = 0
        file_counts = defaultdict(int)
        
        # Duyệt qua tất cả batches
        for batch_dir in self.dataset_root.iterdir():
            if batch_dir.is_dir() and batch_dir.name.startswith("Batch"):
                print(f"\n📂 Processing {batch_dir.name}...")
                
                labels_dir = batch_dir / "labels"
                if not labels_dir.exists():
                    continue
                    
                # Duyệt qua train/val/test
                for split_dir in labels_dir.iterdir():
                    if split_dir.is_dir():
                        print(f"   📁 {split_dir.name}: ", end="")
                        
                        label_files = list(split_dir.glob("*.txt"))
                        print(f"{len(label_files)} files")
                        
                        for label_file in label_files:
                            file_counts[f"{batch_dir.name}_{split_dir.name}"] += 1
                            
                            # Đọc annotations
                            try:
                                with open(label_file, 'r') as f:
                                    for line in f:
                                        if line.strip():
                                            class_id = int(line.strip().split()[0])
                                            class_counts[class_id] += 1
                                            total_annotations += 1
                            except:
                                continue
        
        return class_counts, total_annotations, file_counts
    
    def print_class_analysis(self, class_counts, total_annotations):
        """In phân tích chi tiết về classes"""
        print(f"\n📊 CLASS DISTRIBUTION ANALYSIS")
        print(f"📝 Total annotations: {total_annotations:,}")
        print(f"🏷️ Classes found: {len(class_counts)}")
        
        print(f"\n📋 COMPLETE CLASS MAPPING:")
        print("ID | Official Name        | Count    | % | Traffic Related")
        print("-" * 65)
        
        traffic_total = 0
        for class_id in range(35):
            count = class_counts.get(class_id, 0)
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            
            class_name = self.official_class_names[class_id] if class_id < len(self.official_class_names) else f"Unknown_{class_id}"
            is_traffic = "✅" if class_id in self.traffic_related_classes else "  "
            
            if class_id in self.traffic_related_classes:
                traffic_total += count
                
            print(f"{class_id:2d} | {class_name:20s} | {count:7,d} | {percentage:4.1f}% | {is_traffic}")
        
        traffic_percentage = (traffic_total / total_annotations * 100) if total_annotations > 0 else 0
        print("-" * 65)
        print(f"🚦 TRAFFIC-RELATED CLASSES: {traffic_total:,} annotations ({traffic_percentage:.1f}%)")
        
        # Top traffic classes
        print(f"\n🎯 TOP TRAFFIC CLASSES:")
        traffic_classes_sorted = [(self.traffic_related_classes[cid], class_counts.get(cid, 0)) 
                                 for cid in self.traffic_related_classes.keys()]
        traffic_classes_sorted.sort(key=lambda x: x[1], reverse=True)
        
        for class_name, count in traffic_classes_sorted[:10]:
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"   {class_name:15s}: {count:6,d} ({percentage:4.1f}%)")
    
    def create_revised_11_class_taxonomy(self):
        """Tạo taxonomy 11 classes mới dựa trên findings"""
        print(f"\n🎨 PROPOSED 11-CLASS TAXONOMY FOR TRAFFIC AI")
        print("=" * 60)
        
        # 11 classes optimized cho traffic AI
        taxonomy = {
            0: "pedestrian",       # Person
            1: "bicycle",          # Bicycle  
            2: "motorcycle",       # Motorcycles
            3: "car",              # Car
            4: "bus",              # Bus
            5: "truck",            # Truck
            6: "train",            # Train
            7: "traffic_light",    # Traffic Light
            8: "traffic_sign",     # Stop Sign + other signs
            9: "pothole",          # Path Holes
            10: "infrastructure"   # Barriers + other infrastructure
        }
        
        # Mapping từ Object Detection 35 sang 11-class taxonomy
        od35_to_11_mapping = {
            0: 0,   # Person -> pedestrian
            12: 1,  # Bicycle -> bicycle
            15: 2,  # Motorcycles -> motorcycle  
            24: 3,  # Car -> car
            10: 4,  # Bus -> bus
            14: 5,  # Truck -> truck
            28: 6,  # Train -> train
            20: 7,  # Traffic Light -> traffic_light
            23: 8,  # Stop Sign -> traffic_sign
            26: 9,  # Path Holes -> pothole
            25: 10, # Barriers -> infrastructure
            # Có thể thêm mapping cho các classes khác nếu cần
        }
        
        print("🏗️ 11-Class Taxonomy:")
        for class_id, class_name in taxonomy.items():
            print(f"   {class_id}: {class_name}")
            
        print(f"\n🔗 Object Detection 35 -> 11-Class Mapping:")
        for od35_id, new_id in od35_to_11_mapping.items():
            od35_name = self.official_class_names[od35_id] if od35_id < len(self.official_class_names) else f"Class_{od35_id}"
            new_name = taxonomy[new_id]
            print(f"   {od35_id:2d} ({od35_name:15s}) -> {new_id} ({new_name})")
        
        return taxonomy, od35_to_11_mapping
    
    def save_taxonomy_config(self, taxonomy, od35_mapping):
        """Lưu taxonomy config cho việc merge datasets"""
        config = {
            'project_name': 'Traffic AI YOLOv12 - 11 Classes',
            'description': 'Optimized 11-class taxonomy for traffic object detection',
            
            'classes': list(taxonomy.values()),
            
            'mapping': {
                'object_detection_35': od35_mapping,
                'intersection_flow_5k': {
                    # Sẽ được update sau khi phân tích intersection flow
                    0: 0,  # pedestrian -> pedestrian
                    # ... other mappings
                },
                'vn_traffic_sign': {
                    # Tất cả traffic signs -> traffic_sign  
                    # Sẽ cần phân tích để mapping chi tiết
                },
                'road_issues': {
                    # pothole classes -> pothole
                    # Sẽ mapping từ road issues analysis
                }
            },
            
            'dataset_info': {
                'object_detection_35': {
                    'total_classes': 35,
                    'traffic_related_classes': len(od35_mapping),
                    'official_classes': self.official_class_names
                }
            }
        }
        
        config_path = Path("config/taxonomy_revised_11class.yaml")
        config_path.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n💾 Taxonomy config saved: {config_path}")
        return config_path
    
    def analyze(self):
        """Main analysis function"""
        if not self.dataset_root.exists():
            print(f"❌ Dataset not found: {self.dataset_root}")
            return
            
        # Analyze class distribution
        class_counts, total_annotations, file_counts = self.analyze_class_distribution()
        
        # Print detailed analysis
        self.print_class_analysis(class_counts, total_annotations)
        
        # Create revised taxonomy
        taxonomy, od35_mapping = self.create_revised_11_class_taxonomy()
        
        # Save config
        config_path = self.save_taxonomy_config(taxonomy, od35_mapping)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Found {len(class_counts)} classes with {total_annotations:,} total annotations")
        print(f"🚦 {len(self.traffic_related_classes)} traffic-related classes identified")
        print(f"🎯 11-class taxonomy created and saved")

if __name__ == "__main__":
    analyzer = ObjectDetection35Analyzer()
    analyzer.analyze()