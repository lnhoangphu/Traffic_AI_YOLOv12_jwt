"""
Tool visualize labels để verify xem labels có đúng không
Hiển thị ảnh với bounding boxes và class names
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import json

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "traffic_ai_final_balanced"

CLASS_NAMES = {
    0: "Vehicle",
    1: "Bus",
    2: "Bicycle",
    3: "Person",
    4: "Engine",
    5: "Truck",
    6: "Tricycle",
    7: "Obstacle",
    8: "Pothole",
    9: "Traffic Light",
    10: "Traffic Sign"
}

COLORS = {
    0: (0, 255, 0),      # Vehicle - Green
    1: (255, 0, 0),      # Bus - Blue
    2: (0, 255, 255),    # Bicycle - Yellow
    3: (255, 0, 255),    # Person - Magenta
    4: (128, 128, 0),    # Engine - Teal
    5: (0, 128, 255),    # Truck - Orange
    6: (255, 128, 0),    # Tricycle - Light Blue
    7: (0, 0, 255),      # Obstacle - Red
    8: (128, 0, 128),    # Pothole - Purple
    9: (0, 165, 255),    # Traffic Light - Orange
    10: (255, 255, 0)    # Traffic Sign - Cyan
}

def draw_bbox(img, bbox, class_id, confidence=None):
    """Vẽ bounding box lên ảnh"""
    h, w = img.shape[:2]
    
    # Convert normalized to pixel coordinates
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    
    # Get color and name
    color = COLORS.get(class_id, (255, 255, 255))
    class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label = f"{class_name}"
    if confidence is not None:
        label += f": {confidence:.2f}"
    
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
    
    # Draw text
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return img

def visualize_sample(img_path, label_path, output_path=None, show=True):
    """Visualize một sample"""
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Cannot load image: {img_path}")
        return None
    
    # Load labels
    if not label_path.exists():
        print(f"❌ Label file not found: {label_path}")
        return img
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Draw all bounding boxes
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(float(parts[0]))
        bbox = [float(x) for x in parts[1:5]]
        
        img = draw_bbox(img, bbox, class_id)
    
    # Add image info
    info_text = f"File: {img_path.name} | Objects: {len(lines)}"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), img)
    
    # Show if requested
    if show:
        cv2.imshow("Label Visualization (Press Q to quit, Space for next)", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            cv2.destroyAllWindows()
            return None
    
    return img

def analyze_mislabeling(dataset_path, output_dir=None):
    """Phân tích và visualize các file nghi ngờ mislabeling"""
    dataset_path = Path(dataset_path)
    
    # Load suspicious files from previous analysis
    report_path = PROJECT_ROOT / "label_quality_report.json"
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        suspicious_files = report.get("suspicious_files", [])
        print(f"📊 Found {len(suspicious_files)} suspicious files from analysis")
    else:
        print("⚠️  No quality report found. Run analyze_label_quality.py first.")
        suspicious_files = []
    
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize suspicious files
    print("\n🔍 Visualizing suspicious files...")
    print("Controls: Q=Quit, Space=Next, S=Save")
    
    for i, item in enumerate(suspicious_files[:50]):  # Show first 50
        file_name = item["file"]
        split = item["split"]
        reason = item["reason"]
        
        # Find image and label
        label_path = dataset_path / "labels" / split / file_name
        
        # Find corresponding image
        img_name = file_name.replace(".txt", "")
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            potential = dataset_path / "images" / split / f"{img_name}{ext}"
            if potential.exists():
                img_path = potential
                break
        
        if img_path is None or not label_path.exists():
            continue
        
        print(f"\n[{i+1}/{len(suspicious_files)}] {file_name}")
        print(f"  Reason: {reason}")
        print(f"  Person: {item.get('person_count', 0)}, Vehicle: {item.get('vehicle_count', 0)}")
        
        # Visualize
        output_path = output_dir / f"suspicious_{i:04d}_{file_name.replace('.txt', '.jpg')}" if output_dir else None
        
        result = visualize_sample(img_path, label_path, output_path, show=True)
        
        if result is None:  # User pressed Q
            break
    
    cv2.destroyAllWindows()
    print("\n✅ Visualization complete!")
    
    if output_dir:
        print(f"📁 Saved visualizations to: {output_dir}")

def visualize_random_samples(dataset_path, split="train", n_samples=10):
    """Visualize random samples để kiểm tra general quality"""
    dataset_path = Path(dataset_path)
    
    labels_dir = dataset_path / "labels" / split
    images_dir = dataset_path / "images" / split
    
    if not labels_dir.exists():
        print(f"❌ {split} split not found")
        return
    
    # Get random samples
    label_files = list(labels_dir.glob("*.txt"))
    samples = random.sample(label_files, min(n_samples, len(label_files)))
    
    print(f"\n🎲 Visualizing {len(samples)} random samples from {split}...")
    print("Controls: Q=Quit, Space=Next")
    
    for i, label_file in enumerate(samples):
        # Find image
        img_name = label_file.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            potential = images_dir / f"{img_name}{ext}"
            if potential.exists():
                img_path = potential
                break
        
        if img_path is None:
            continue
        
        print(f"\n[{i+1}/{len(samples)}] {img_path.name}")
        
        result = visualize_sample(img_path, label_file, show=True)
        
        if result is None:
            break
    
    cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize dataset labels")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Dataset path")
    parser.add_argument("--mode", choices=["suspicious", "random", "file"], default="suspicious",
                       help="Visualization mode")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                       help="Dataset split")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples (for random mode)")
    parser.add_argument("--file", help="Specific file to visualize (for file mode)")
    parser.add_argument("--output", help="Output directory to save visualizations")
    
    args = parser.parse_args()
    
    if args.mode == "suspicious":
        analyze_mislabeling(args.dataset, args.output)
    
    elif args.mode == "random":
        visualize_random_samples(args.dataset, args.split, args.n_samples)
    
    elif args.mode == "file" and args.file:
        dataset_path = Path(args.dataset)
        label_path = dataset_path / "labels" / args.split / args.file
        
        img_name = Path(args.file).stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            potential = dataset_path / "images" / args.split / f"{img_name}{ext}"
            if potential.exists():
                img_path = potential
                break
        
        if img_path and label_path.exists():
            visualize_sample(img_path, label_path, show=True)
        else:
            print(f"❌ File not found: {args.file}")

if __name__ == "__main__":
    main()
