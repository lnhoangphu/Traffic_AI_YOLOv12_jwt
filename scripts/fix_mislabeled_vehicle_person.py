"""
Tool tự động phát hiện và sửa labels bị gán nhầm giữa Vehicle và Person
Sử dụng heuristics và pretrained model để re-label
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import shutil
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "traffic_ai_final_balanced"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "traffic_ai_final_balanced_fixed"

# Load pretrained YOLO model để verify
PRETRAINED_MODEL = PROJECT_ROOT / "yolo12n.pt"

class LabelFixer:
    def __init__(self, dataset_path, output_path, use_pretrained=True):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.use_pretrained = use_pretrained
        
        if use_pretrained and PRETRAINED_MODEL.exists():
            print(f"📦 Loading pretrained model: {PRETRAINED_MODEL}")
            self.model = YOLO(str(PRETRAINED_MODEL))
        else:
            self.model = None
            print("⚠️  Running in heuristic-only mode")
        
        self.stats = {
            "total_files": 0,
            "fixed_files": 0,
            "person_to_vehicle": 0,
            "vehicle_to_person": 0,
            "unchanged": 0,
            "errors": []
        }
    
    def analyze_bbox_features(self, bbox, img_shape):
        """Phân tích đặc điểm của bbox để detect mislabel"""
        h, w = img_shape[:2]
        
        # Convert normalized coords to pixels
        x_center, y_center, width, height = bbox[1:]
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        
        # Calculate features
        aspect_ratio = height / width if width > 0 else 1.0
        area = width * height
        relative_size = area / (h * w)
        
        # Position features
        is_on_road = y_center > h * 0.3  # Below 30% of image
        is_large = width > w * 0.1 or height > h * 0.15
        
        return {
            "aspect_ratio": aspect_ratio,
            "area": area,
            "relative_size": relative_size,
            "is_on_road": is_on_road,
            "is_large": is_large,
            "width": width,
            "height": height
        }
    
    def is_likely_vehicle(self, class_id, features, context):
        """Heuristic để phát hiện object có thể là vehicle"""
        
        # Nếu đã là Vehicle, giữ nguyên
        if class_id == 0:  # Vehicle
            return True
        
        # Nếu là Person, check xem có phải vehicle không
        if class_id == 3:  # Person
            # Large bounding box + on road + aspect ratio gần 1 -> likely vehicle
            if (features["is_large"] and 
                features["is_on_road"] and 
                0.5 < features["aspect_ratio"] < 2.0):
                
                # Nếu có nhiều "person" boxes gần nhau -> có thể là traffic
                if context["nearby_persons"] > 5:
                    return True
                
                # Box rất lớn -> xe chứ không phải người
                if features["relative_size"] > 0.05:  # > 5% ảnh
                    return True
        
        return False
    
    def is_likely_person(self, class_id, features):
        """Heuristic để phát hiện object có thể là person"""
        
        # Nếu đã là Person, giữ nguyên
        if class_id == 3:
            return True
        
        # Nếu là Vehicle, check xem có phải person không
        if class_id == 0:  # Vehicle
            # Small box + vertical aspect ratio -> likely person
            if (not features["is_large"] and 
                features["aspect_ratio"] > 1.3 and
                features["relative_size"] < 0.01):  # < 1% ảnh
                return True
        
        return False
    
    def get_context(self, labels, current_idx):
        """Get context về surrounding objects"""
        context = {
            "total_objects": len(labels),
            "person_count": sum(1 for l in labels if int(float(l[0])) == 3),
            "vehicle_count": sum(1 for l in labels if int(float(l[0])) == 0),
            "nearby_persons": 0
        }
        
        # Count nearby persons (simple spatial check)
        if current_idx < len(labels):
            current = labels[current_idx]
            cx, cy = float(current[1]), float(current[2])
            
            for i, label in enumerate(labels):
                if i == current_idx:
                    continue
                if int(float(label[0])) == 3:  # Person
                    ox, oy = float(label[1]), float(label[2])
                    dist = ((cx - ox)**2 + (cy - oy)**2)**0.5
                    if dist < 0.2:  # Trong bán kính 20% ảnh
                        context["nearby_persons"] += 1
        
        return context
    
    def use_pretrained_verification(self, img_path, bbox):
        """Dùng pretrained model để verify class"""
        if self.model is None:
            return None
        
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            h, w = img.shape[:2]
            
            # Convert normalized bbox to pixels
            x_center, y_center, width, height = bbox[1:]
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Crop region
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            crop = img[y1:y2, x1:x2]
            
            # Run detection on crop
            results = self.model.predict(crop, verbose=False, conf=0.1)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get top prediction
                top_box = results[0].boxes[0]
                predicted_class = int(top_box.cls)
                confidence = float(top_box.conf)
                
                # Map COCO classes to our classes
                # COCO: 0=person, 2=car, 5=bus, 7=truck, 1=bicycle, 3=motorcycle
                coco_to_our = {
                    0: 3,  # person -> Person
                    2: 0,  # car -> Vehicle
                    5: 1,  # bus -> Bus
                    7: 5,  # truck -> Truck
                    1: 2,  # bicycle -> Bicycle
                    3: 4,  # motorcycle -> Engine
                }
                
                our_class = coco_to_our.get(predicted_class)
                
                return {
                    "predicted_class": our_class,
                    "confidence": confidence,
                    "coco_class": predicted_class
                }
        
        except Exception as e:
            print(f"Error in pretrained verification: {e}")
            return None
    
    def fix_label_file(self, label_path, img_path, split):
        """Fix một label file"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return False
            
            # Load image để get shape
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            fixed_lines = []
            modified = False
            
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    fixed_lines.append(line)
                    continue
                
                class_id = int(float(parts[0]))
                bbox = [float(x) for x in parts]
                
                # Analyze features
                features = self.analyze_bbox_features(bbox, img.shape)
                context = self.get_context([l.strip().split() for l in lines], idx)
                
                # Decision logic
                new_class_id = class_id
                reason = "unchanged"
                
                # Rule 1: Heuristic-based correction
                if class_id == 3:  # Person
                    if self.is_likely_vehicle(class_id, features, context):
                        new_class_id = 0  # Change to Vehicle
                        reason = "heuristic_person_to_vehicle"
                        self.stats["person_to_vehicle"] += 1
                        modified = True
                
                elif class_id == 0:  # Vehicle
                    if self.is_likely_person(class_id, features):
                        new_class_id = 3  # Change to Person
                        reason = "heuristic_vehicle_to_person"
                        self.stats["vehicle_to_person"] += 1
                        modified = True
                
                # Rule 2: Pretrained model verification (if enabled)
                if self.model is not None and new_class_id != class_id:
                    verification = self.use_pretrained_verification(img_path, bbox)
                    if verification and verification["confidence"] > 0.5:
                        # Use pretrained prediction if confident
                        if verification["predicted_class"] is not None:
                            new_class_id = verification["predicted_class"]
                            reason = f"pretrained_{verification['confidence']:.2f}"
                
                # Build fixed line
                fixed_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                fixed_lines.append(fixed_line)
            
            # Write fixed labels
            if modified:
                output_label_path = self.output_path / "labels" / split / label_path.name
                output_label_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_label_path, 'w') as f:
                    f.writelines(fixed_lines)
                
                # Copy image
                output_img_path = self.output_path / "images" / split / img_path.name
                output_img_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, output_img_path)
                
                return True
            
            return False
        
        except Exception as e:
            self.stats["errors"].append(f"{label_path.name}: {str(e)}")
            return False
    
    def process_dataset(self):
        """Process toàn bộ dataset"""
        print("🔧 FIXING MISLABELED DATA")
        print("="*60)
        
        for split in ["train", "val", "test"]:
            labels_dir = self.dataset_path / "labels" / split
            images_dir = self.dataset_path / "images" / split
            
            if not labels_dir.exists():
                continue
            
            label_files = list(labels_dir.glob("*.txt"))
            print(f"\n📂 Processing {split} split: {len(label_files)} files")
            
            for label_file in tqdm(label_files, desc=f"Fixing {split}"):
                self.stats["total_files"] += 1
                
                # Find corresponding image
                img_name = label_file.stem.replace(".txt", "")
                img_file = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    potential = images_dir / f"{img_name}{ext}"
                    if potential.exists():
                        img_file = potential
                        break
                
                if img_file is None:
                    continue
                
                # Fix labels
                if self.fix_label_file(label_file, img_file, split):
                    self.stats["fixed_files"] += 1
        
        # Copy data.yaml
        if (self.dataset_path / "data.yaml").exists():
            shutil.copy2(
                self.dataset_path / "data.yaml",
                self.output_path / "data.yaml"
            )
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """In thống kê"""
        print("\n" + "="*60)
        print("📊 FIXING STATISTICS")
        print("="*60)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Files modified: {self.stats['fixed_files']}")
        print(f"Person → Vehicle: {self.stats['person_to_vehicle']}")
        print(f"Vehicle → Person: {self.stats['vehicle_to_person']}")
        print(f"Unchanged: {self.stats['total_files'] - self.stats['fixed_files']}")
        
        if self.stats["errors"]:
            print(f"\n⚠️  Errors: {len(self.stats['errors'])}")
            for err in self.stats["errors"][:5]:
                print(f"  - {err}")
        
        # Save report
        report_path = self.output_path / "fixing_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n💾 Report saved to: {report_path}")
        print(f"📁 Fixed dataset saved to: {self.output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix mislabeled Vehicle/Person")
    parser.add_argument("--input", default=str(DATASET_PATH), help="Input dataset path")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output dataset path")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained model verification")
    parser.add_argument("--dry-run", action="store_true", help="Only analyze, don't fix")
    
    args = parser.parse_args()
    
    fixer = LabelFixer(
        dataset_path=args.input,
        output_path=args.output,
        use_pretrained=not args.no_pretrained
    )
    
    fixer.process_dataset()
    
    print("\n✅ Done! Review fixed dataset before training.")
    print(f"\nTo train with fixed dataset:")
    print(f"  python training/train_11class_final.py --data {args.output}/data.yaml")

if __name__ == "__main__":
    main()
