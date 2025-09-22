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
    """Tạo taxonomy hoàn chỉnh theo đề xuất của user - 11 classes tối ưu"""
    
    # 11 classes theo đề xuất của user
    classes = [
        "Vehicle",         # 0 - Các phương tiện giao thông khác (car, vehicle chung)
        "Bus",             # 1 - Xe buýt, xe khách
        "Bicycle",         # 2 - Xe đạp các loại
        "Person",          # 3 - Người (đi bộ, ngồi trên xe, tất cả người)
        "Engine",          # 4 - Xe 2 bánh có động cơ (motorcycle, scooter)
        "Truck",           # 5 - Xe tải các loại
        "Tricycle",        # 6 - Xe 3 bánh
        "Obstacle",        # 7 - Vật cản, chướng ngại vật
        "Pothole",         # 8 - Ổ gà, hư hỏng mặt đường
        "Traffic Light",   # 9 - Đèn giao thông
        "Traffic Sign"     # 10 - Biển báo giao thông (29 biển VN + các biển khác)
    ]
    
    # Mapping cho từng dataset theo đề xuất của user
    mapping = {
        # Intersection Flow 5K - Mapping chính xác theo classes hiện tại
        'intersection_flow_5k': {
            0: 3,  # pedestrian -> Person
            1: 2,  # bicycle -> Bicycle
            2: 0,  # vehicle -> Vehicle
            3: 0,  # car -> Vehicle (gộp vào Vehicle)
            4: 5,  # truck -> Truck
            5: 1,  # bus -> Bus
            6: 4,  # engine -> Engine
            7: 6,  # tricycle -> Tricycle
        },
        
        # Object Detection 35 - Traffic-related classes mapping
        'object_detection_35': {
            0: 3,   # Person -> Person
            10: 1,  # Bus -> Bus
            12: 2,  # Bicycle -> Bicycle
            14: 5,  # Truck -> Truck
            15: 4,  # Motorcycles -> Engine
            20: 9,  # Traffic Light -> Traffic Light
            23: 10, # Stop Sign -> Traffic Sign
            24: 0,  # Car -> Vehicle
            25: 7,  # Barriers -> Obstacle
            26: 8,  # Path Holes -> Pothole
            28: 0,  # Train -> Vehicle (hoặc có thể tách riêng nếu cần)
        },
        
        # VN Traffic Sign Dataset - Tất cả signs -> Traffic Sign
        'vn_traffic_sign': {
            'all_classes': 10  # All 29 traffic signs -> Traffic Sign
        },
        
        # Road Issues Dataset
        'road_issues': {
            0: 10,  # broken_road_sign -> Traffic Sign
            4: 8,   # pothole -> Pothole
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
        'project_name': 'Traffic AI YOLOv12 - User-Specified 11-Class Taxonomy',
        'version': '3.0',
        'description': 'User-optimized 11-class taxonomy: Vehicle, Bus, Bicycle, Person, Engine, Truck, Tricycle, Obstacle, Pothole, Traffic Light, Traffic Sign',
        'created_date': '2025-09-20',
        
        'classes': classes,
        'num_classes': len(classes),
        
        'class_descriptions': {
            0: 'Vehicle - Các phương tiện giao thông khác (cars, vehicles chung không thuộc các loại riêng biệt)',
            1: 'Bus - Xe buýt, xe khách các loại',
            2: 'Bicycle - Xe đạp tất cả các loại',
            3: 'Person - Tất cả người (pedestrian, passenger, người trên xe, người đi bộ)',
            4: 'Engine - Xe 2 bánh có động cơ (motorcycle, scooter, xe máy)',
            5: 'Truck - Xe tải các loại (truck, van, xe chở hàng)',
            6: 'Tricycle - Xe 3 bánh (tricycle, xe lam)',
            7: 'Obstacle - Vật cản, chướng ngại vật trên đường',
            8: 'Pothole - Ổ gà, hư hỏng mặt đường',
            9: 'Traffic Light - Đèn giao thông',
            10: 'Traffic Sign - Tất cả biển báo giao thông (29 loại VN + các biển khác)'
        },
        
        'mapping': mapping,
        'dataset_info': dataset_info,
        
        'training_strategy': {
            'data_balance': {
                'high_priority': [0, 3, 9, 10],  # Vehicle, Person, Traffic Light, Traffic Sign
                'medium_priority': [1, 2, 4, 5], # Bus, Bicycle, Engine, Truck
                'low_priority': [6, 7, 8]        # Tricycle, Obstacle, Pothole
            },
            'augmentation_focus': [6, 7, 8],     # Classes with fewer samples
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