"""
Test script cho FastAPI service.
Kiểm tra các endpoints và tính năng detection.
"""

import requests
import json
from pathlib import Path
import base64
import time

API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        if response.status_code == 200:
            print("✅ Health check: OK")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint: OK")
            print(f"   Message: {data.get('message')}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_detect_endpoint():
    """Test detection endpoint với ảnh mẫu"""
    
    # Tìm ảnh test
    REPO_ROOT = Path(__file__).resolve().parents[1]
    test_img_dir = REPO_ROOT / "data" / "traffic" / "images" / "test"
    
    if not test_img_dir.exists():
        print("❌ Không tìm thấy thư mục test images")
        return False
    
    test_images = list(test_img_dir.glob("*.jpg"))[:3]  # Test 3 ảnh
    if not test_images:
        print("❌ Không tìm thấy ảnh test")
        return False
    
    print(f"Testing detection với {len(test_images)} ảnh...")
    
    success_count = 0
    for img_path in test_images:
        try:
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE_URL}/detect",
                    files=files,
                    timeout=30
                )
                inference_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    objects = data.get('objects', [])
                    
                    print(f"✅ {img_path.name}: {len(objects)} objects detected ({inference_time:.2f}s)")
                    
                    # In chi tiết objects
                    for obj in objects[:3]:  # Chỉ in 3 object đầu
                        print(f"   - {obj['label']}: {obj['confidence']:.2f}")
                    
                    success_count += 1
                    
                else:
                    print(f"❌ {img_path.name}: HTTP {response.status_code}")
                    print(f"   Response: {response.text}")
                    
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")
    
    if success_count == len(test_images):
        print("✅ Tất cả test detection thành công")
        return True
    else:
        print(f"❌ {success_count}/{len(test_images)} test detection thành công")
        return False

def test_invalid_requests():
    """Test các trường hợp invalid request"""
    
    print("Testing invalid requests...")
    
    # 1. Test không có file
    try:
        response = requests.post(f"{API_BASE_URL}/detect", timeout=5)
        if response.status_code == 422:  # Unprocessable Entity
            print("✅ No file test: Correctly rejected")
        else:
            print(f"❌ No file test: Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"❌ No file test error: {e}")
    
    # 2. Test file không phải ảnh
    try:
        fake_file = b"This is not an image"
        files = {'file': ('test.txt', fake_file, 'text/plain')}
        
        response = requests.post(
            f"{API_BASE_URL}/detect",
            files=files,
            timeout=5
        )
        
        if response.status_code == 400:  # Bad Request
            print("✅ Invalid file type test: Correctly rejected")
        else:
            print(f"❌ Invalid file type test: Expected 400, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Invalid file type test error: {e}")

def benchmark_performance():
    """Benchmark hiệu suất API"""
    
    print("\n=== PERFORMANCE BENCHMARK ===")
    
    REPO_ROOT = Path(__file__).resolve().parents[1] 
    test_img_dir = REPO_ROOT / "data" / "traffic" / "images" / "test"
    
    if not test_img_dir.exists():
        print("❌ Không có ảnh để benchmark")
        return
    
    test_images = list(test_img_dir.glob("*.jpg"))[:10]  # Test 10 ảnh
    if not test_images:
        print("❌ Không tìm thấy ảnh test")
        return
    
    inference_times = []
    detection_counts = []
    
    for img_path in test_images:
        try:
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post(f"{API_BASE_URL}/detect", files=files, timeout=30)
                inference_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    objects = data.get('objects', [])
                    
                    inference_times.append(inference_time)
                    detection_counts.append(len(objects))
                    
        except Exception as e:
            print(f"Benchmark error với {img_path.name}: {e}")
    
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        avg_detections = sum(detection_counts) / len(detection_counts)
        
        print(f"Processed {len(inference_times)} images:")
        print(f"  Average inference time: {avg_time:.3f}s")
        print(f"  Min inference time: {min_time:.3f}s")
        print(f"  Max inference time: {max_time:.3f}s")
        print(f"  Average detections per image: {avg_detections:.1f}")
        print(f"  Throughput: {1/avg_time:.1f} images/second")

def main():
    """Main test function"""
    
    print("=== TESTING TRAFFIC AI API SERVICE ===")
    print(f"API URL: {API_BASE_URL}")
    print("\nCHÚ Ý: Đảm bảo API service đang chạy!")
    print("Để start service: uvicorn src.ai_service.main:app --host 0.0.0.0 --port 8000\n")
    
    # Test các endpoints
    all_passed = True
    
    print("1. Testing basic endpoints...")
    if not test_health_endpoint():
        all_passed = False
    
    if not test_root_endpoint():
        all_passed = False
    
    print("\n2. Testing detection endpoint...")
    if not test_detect_endpoint():
        all_passed = False
    
    print("\n3. Testing invalid requests...")
    test_invalid_requests()
    
    print("\n4. Performance benchmark...")
    benchmark_performance()
    
    print("\n=== TEST SUMMARY ===")
    if all_passed:
        print("✅ Tất cả test cơ bản đều PASS")
    else:
        print("❌ Một số test FAILED - kiểm tra lại service")
    
    print("\n💡 TIPS:")
    print("- Để cải thiện performance, cân nhắc deploy trên GPU")
    print("- Có thể tăng batch size nếu có đủ memory")
    print("- Monitor API trong production với logging và metrics")

if __name__ == "__main__":
    main()