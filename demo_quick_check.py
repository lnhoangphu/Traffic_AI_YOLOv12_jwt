#!/usr/bin/env python3
"""
Quick demo script to verify dataset creation and show statistics
"""

import os
import yaml
from pathlib import Path

def load_yaml(file_path):
    """Load YAML file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def count_files_in_directory(directory):
    """Count files in directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def analyze_dataset(dataset_path):
    """Analyze dataset structure and statistics"""
    print(f"\n📊 Analyzing Dataset: {os.path.basename(dataset_path)}")
    print("=" * 60)
    
    # Check if data.yaml exists
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(data_yaml_path):
        config = load_yaml(data_yaml_path)
        print(f"📝 Classes: {config['nc']} classes")
        for i, class_name in enumerate(config['names']):
            print(f"   {i}: {class_name}")
    
    # Count files in each split
    splits = ['train', 'val', 'test']
    total_images = 0
    total_labels = 0
    
    for split in splits:
        images_dir = os.path.join(dataset_path, 'images', split)
        labels_dir = os.path.join(dataset_path, 'labels', split)
        
        image_count = count_files_in_directory(images_dir)
        label_count = count_files_in_directory(labels_dir)
        
        total_images += image_count
        total_labels += label_count
        
        print(f"📁 {split.capitalize()}: {image_count} images, {label_count} labels")
    
    print(f"📈 Total: {total_images} images, {total_labels} labels")
    
    return total_images, total_labels

def main():
    """Main demo function"""
    print("🚀 Traffic AI Dataset Quick Check Demo")
    print("=" * 60)
    
    # Base directories
    datasets_dir = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets"
    
    # Dataset paths
    balanced_path = os.path.join(datasets_dir, "traffic_ai_balanced_11class")
    imbalanced_path = os.path.join(datasets_dir, "traffic_ai_imbalanced_11class")
    
    # Analyze datasets
    if os.path.exists(balanced_path):
        balanced_total, _ = analyze_dataset(balanced_path)
    else:
        print(f"❌ Balanced dataset not found: {balanced_path}")
        balanced_total = 0
    
    if os.path.exists(imbalanced_path):
        imbalanced_total, _ = analyze_dataset(imbalanced_path)
    else:
        print(f"❌ Imbalanced dataset not found: {imbalanced_path}")
        imbalanced_total = 0
    
    # Summary comparison
    print("\n🔍 Dataset Comparison Summary")
    print("=" * 60)
    print(f"📊 Balanced Dataset:   {balanced_total:,} images")
    print(f"📊 Imbalanced Dataset: {imbalanced_total:,} images")
    
    if balanced_total > 0 and imbalanced_total > 0:
        ratio = imbalanced_total / balanced_total
        print(f"📈 Size Ratio: {ratio:.1f}x (Imbalanced is {ratio:.1f}x larger)")
    
    # Check taxonomy file
    taxonomy_path = os.path.join(r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\config", "taxonomy_complete_11class.yaml")
    if os.path.exists(taxonomy_path):
        print(f"\n✅ Taxonomy configuration found: {taxonomy_path}")
        taxonomy = load_yaml(taxonomy_path)
        print(f"📝 Target classes: {taxonomy.get('target_classes', 'Not found')}")
    else:
        print(f"\n❌ Taxonomy configuration not found: {taxonomy_path}")
    
    print("\n🎉 Quick check completed!")
    print("Ready for training and research! 🚀")

if __name__ == "__main__":
    main()