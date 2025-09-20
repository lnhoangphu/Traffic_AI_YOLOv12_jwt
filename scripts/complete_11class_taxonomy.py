"""
Complete 11-Class Taxonomy Configuration for Traffic AI YOLOv12
Hoàn thiện mapping cho tất cả 4 datasets với 11 classes optimized

Classes:
0: pedestrian, 1: bicycle, 2: motorcycle, 3: car, 4: bus, 5: truck, 
6: train, 7: traffic_light, 8: traffic_sign, 9: pothole, 10: infrastructure

Usage: python scripts/complete_11class_taxonomy.py
"""

import yaml
from pathlib import Path

def create_complete_taxonomy():
    """Tạo taxonomy hoàn chỉnh cho tất cả 4 datasets"""
    
    # 11 classes chuẩn
    classes = [
        "pedestrian",      # 0 - Người đi bộ
        "bicycle",         # 1 - Xe đạp  
        "motorcycle",      # 2 - Xe máy
        "car",             # 3 - Xe hơi
        "bus",             # 4 - Xe bus
        "truck",           # 5 - Xe tải
        "train",           # 6 - Xe lửa
        "traffic_light",   # 7 - Đèn giao thông
        "traffic_sign",    # 8 - Biển báo giao thông
        "pothole",         # 9 - Ổ gà đường
        "infrastructure"   # 10 - Hạ tầng giao thông
    ]
    
    # Mapping cho từng dataset
    mapping = {
        # Object Detection 35 Classes (VisionGuard dataset)
        'object_detection_35': {
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
            # Other classes are ignored (không map)
        },
        
        # Intersection Flow 5K (cần phân tích thêm)
        'intersection_flow_5k': {
            # Dựa trên analysis trước đây:
            # pedestrian, bicycle, vehicle, car, truck, bus, engine, tricycle
            0: 0,  # pedestrian -> pedestrian
            1: 1,  # bicycle -> bicycle
            2: 3,  # vehicle -> car (generic vehicle)
            3: 3,  # car -> car
            4: 5,  # truck -> truck
            5: 4,  # bus -> bus
            6: 2,  # engine -> motorcycle (engine/motorbike)
            7: 2,  # tricycle -> motorcycle (3-wheel vehicle)
        },
        
        # VN Traffic Sign Dataset
        'vn_traffic_sign': {
            # Tất cả traffic signs map to traffic_sign class
            # Sẽ cần loop qua tất cả 29 classes và map về class 8
            # Format sẽ được update sau khi phân tích chi tiết
            'all_classes': 8  # All traffic signs -> traffic_sign
        },
        
        # Road Issues Dataset  
        'road_issues': {
            # Dựa trên class mapping từ convert_road_issues.py:
            4: 9,  # pothole -> pothole
            0: 8,  # broken_road_sign -> traffic_sign
            # Other road issues không có trong 11-class taxonomy
        }
    }
    
    # Dataset statistics và info
    dataset_info = {
        'object_detection_35': {
            'total_classes': 35,
            'traffic_classes_used': 11,
            'total_annotations': 23043,
            'traffic_annotations': 7503,
            'description': 'VisionGuard dataset with 35 object classes for blind assistance'
        },
        'intersection_flow_5k': {
            'total_classes': 8,
            'traffic_classes_used': 6,
            'estimated_annotations': 400000,  # Cần verify
            'description': 'Traffic intersection surveillance data'
        },
        'vn_traffic_sign': {
            'total_classes': 29,
            'traffic_classes_used': 1,  # All map to traffic_sign
            'estimated_annotations': 2800,
            'description': 'Vietnamese traffic signs dataset'
        },
        'road_issues': {
            'total_classes': 7,
            'traffic_classes_used': 2,  # pothole + broken_road_sign
            'total_images': 9660,
            'description': 'Road infrastructure problems dataset'
        }
    }
    
    # Complete taxonomy config
    taxonomy_config = {
        'project_name': 'Traffic AI YOLOv12 - Complete 11-Class Taxonomy',
        'version': '2.0',
        'description': 'Optimized 11-class taxonomy for comprehensive traffic object detection',
        'created_date': '2025-09-20',
        
        'classes': classes,
        'num_classes': len(classes),
        
        'class_descriptions': {
            0: 'Pedestrian - Người đi bộ trên đường',
            1: 'Bicycle - Xe đạp các loại',
            2: 'Motorcycle - Xe máy, xe 3 bánh',  
            3: 'Car - Xe hơi, xe con',
            4: 'Bus - Xe buýt, xe khách',
            5: 'Truck - Xe tải các loại',
            6: 'Train - Xe lửa, tàu hỏa',
            7: 'Traffic Light - Đèn giao thông',
            8: 'Traffic Sign - Biển báo giao thông',
            9: 'Pothole - Ổ gà, hư hỏng mặt đường',
            10: 'Infrastructure - Hạ tầng giao thông (barriers, etc.)'
        },
        
        'mapping': mapping,
        'dataset_info': dataset_info,
        
        'training_strategy': {
            'data_balance': {
                'high_priority': [0, 3, 7, 8],  # pedestrian, car, traffic_light, traffic_sign
                'medium_priority': [1, 2, 4, 5], # bicycle, motorcycle, bus, truck
                'low_priority': [6, 9, 10]       # train, pothole, infrastructure
            },
            'augmentation_focus': [2, 6, 7, 9, 10],  # Classes with fewer samples
            'class_weights': 'inverse_frequency'
        }
    }
    
    return taxonomy_config

def save_taxonomy_config(config):
    """Lưu taxonomy config"""
    config_path = Path("config/taxonomy_complete_11class.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ Complete taxonomy saved: {config_path}")
    return config_path

def print_taxonomy_summary(config):
    """In summary của taxonomy"""
    print("🎯 COMPLETE 11-CLASS TRAFFIC AI TAXONOMY")
    print("=" * 60)
    
    print(f"📊 Classes: {config['num_classes']}")
    for i, class_name in enumerate(config['classes']):
        desc = config['class_descriptions'][i]
        print(f"   {i:2d}: {class_name:15s} - {desc}")
    
    print(f"\n📂 Datasets covered:")
    for dataset, info in config['dataset_info'].items():
        print(f"   ✅ {dataset:20s}: {info['traffic_classes_used']:2d}/{info['total_classes']:2d} classes used")
    
    print(f"\n🔗 Mapping summary:")
    for dataset, mapping_info in config['mapping'].items():
        if isinstance(mapping_info, dict):
            mapped_classes = len(mapping_info)
            print(f"   {dataset:20s}: {mapped_classes} classes mapped")
        else:
            print(f"   {dataset:20s}: All classes -> traffic_sign")

def main():
    print("🚀 CREATING COMPLETE 11-CLASS TAXONOMY")
    print("=" * 60)
    
    # Create complete taxonomy
    config = create_complete_taxonomy()
    
    # Print summary
    print_taxonomy_summary(config)
    
    # Save config
    config_path = save_taxonomy_config(config)
    
    print(f"\n✅ TAXONOMY CREATION COMPLETE!")
    print(f"💾 Config saved: {config_path}")
    print(f"\n📋 NEXT STEPS:")
    print("   1. Update dataset conversion scripts")
    print("   2. Create unified dataset merger")
    print("   3. Start training with 11-class taxonomy")

if __name__ == "__main__":
    main()