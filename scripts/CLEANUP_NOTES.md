# Traffic AI YOLOv12 - Scripts Cleanup Summary
# =============================================================================

## ACTIVE SCRIPTS (Keep these):

### Core Processing Scripts:
- `analyze_object_detection_35_correct.py` - Analyze Object Detection 35 dataset
- `complete_11class_taxonomy.py` - Generate 11-class taxonomy configuration
- `convert_road_issues.py` - Convert Road Issues dataset to YOLO format
- `organize_object_detection_35_keep_original.py` - Organize Object Detection 35 (35 classes)
- `merge_datasets_final_correct.py` - Final merger for 11-class taxonomy ⭐

### Utilities:
- `download_kaggle.ps1/sh` - Download datasets from Kaggle
- `download_yolo12n.py` - Download YOLO model weights
- `quick_check_yolo12n.py` - Quick model verification
- `verify_converted_datasets.py` - Verify dataset conversions
- `train_balanced_vs_imbalanced_fixed.py` - Training script

## DEPRECATED SCRIPTS (Should be removed):

### Old/Incorrect Versions:
- `convert_object_detection_35.py` - OLD: Wrong class mapping (cutlery)
- `convert_object_detection_35_11class.py` - OLD: Wrong approach (convert before merge)
- `create_taxonomy_mapping.py` - OLD: Superseded by complete_11class_taxonomy.py
- `organize_object_detection_35.py` - OLD: Wrong class mapping

### Duplicate/Outdated Mergers:
- `merge_datasets_fixed.py` - OLD: Outdated approach
- `merge_datasets_optimized.py` - OLD: Outdated approach  
- `merge_final_11class.py` - DUPLICATE: Same as merge_datasets_final_correct.py

### Temporary Downloads:
- `download_object_detection_35_correct.py` - TEMP: One-time download script

## CLEANUP COMMANDS:
```powershell
# Remove deprecated scripts
Remove-Item scripts/convert_object_detection_35.py -ErrorAction SilentlyContinue
Remove-Item scripts/convert_object_detection_35_11class.py -ErrorAction SilentlyContinue  
Remove-Item scripts/create_taxonomy_mapping.py -ErrorAction SilentlyContinue
Remove-Item scripts/organize_object_detection_35.py -ErrorAction SilentlyContinue
Remove-Item scripts/merge_datasets_fixed.py -ErrorAction SilentlyContinue
Remove-Item scripts/merge_datasets_optimized.py -ErrorAction SilentlyContinue
Remove-Item scripts/merge_final_11class.py -ErrorAction SilentlyContinue
Remove-Item scripts/download_object_detection_35_correct.py -ErrorAction SilentlyContinue
```