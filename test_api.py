#!/usr/bin/env python3
"""
Script test nhanh API Traffic AI
"""

import requests
import json
import time
from pathlib import Path
from PIL import Image
import io

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_health(self):
        """Test endpoint health"""
        print("🏥 Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health OK: {data}")
                return True
            else:
                print(f"❌ Health failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health error: {e}")
            return False
    
    def test_model_info(self):
        """Test model info endpoint"""
        print("\n📋 Testing model info...")
        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model Info:")
                print(f"   Type: {data.get('model_type')}")
                print(f"   Classes: {data.get('num_classes')}")
                print(f"   Path: {data.get('model_path')}")
                if 'performance' in data:
                    print(f"   Performance: {data['performance']}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
    
    def test_benchmark(self, samples=5):
        """Test benchmark endpoint"""
        print(f"\n⚡ Testing benchmark with {samples} samples...")
        try:
            response = requests.post(
                f"{self.base_url}/test/benchmark?num_samples={samples}", 
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                perf = data.get('performance', {})
                print(f"✅ Benchmark Results:")
                print(f"   Avg Time: {perf.get('avg_inference_time_ms')}ms")
                print(f"   Throughput: {perf.get('throughput_fps')} FPS")
                print(f"   Status: {data.get('status')}")
                return True
            else:
                print(f"❌ Benchmark failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            return False
    
    def test_detection(self):
        """Test detection với ảnh test"""
        print("\n🔍 Testing detection...")
        try:
            # Tạo ảnh test đơn giản
            test_img = Image.new('RGB', (640, 640), color=(100, 150, 200))
            
            # Vẽ một số hình để model có thể detect
            from PIL import ImageDraw
            draw = ImageDraw.Draw(test_img)
            
            # Vẽ một số hình chữ nhật (giả lập xe, người, etc.)
            draw.rectangle([100, 100, 200, 200], fill=(255, 0, 0))  # Red box
            draw.rectangle([300, 300, 400, 450], fill=(0, 255, 0))  # Green box
            draw.ellipse([500, 150, 580, 230], fill=(255, 255, 0))  # Yellow circle
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            test_img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Send request
            files = {'file': ('test.jpg', img_bytes.getvalue(), 'image/jpeg')}
            response = requests.post(f"{self.base_url}/detect", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {})
                print(f"✅ Detection Results:")
                print(f"   Success: {data.get('success')}")
                print(f"   Objects found: {results.get('total_objects', 0)}")
                print(f"   High confidence: {results.get('high_confidence_objects', 0)}")
                print(f"   Inference time: {data.get('inference', {}).get('time_ms')}ms")
                
                if results.get('objects'):
                    print(f"   Detected classes: {results.get('class_distribution', {})}")
                
                return True
            else:
                print(f"❌ Detection failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return False
    
    def test_all(self):
        """Chạy tất cả tests"""
        print("🧪 STARTING API TESTS")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health),
            ("Model Info", self.test_model_info),
            ("Benchmark", lambda: self.test_benchmark(3)),
            ("Detection", self.test_detection)
        ]
        
        results = []
        for test_name, test_func in tests:
            start_time = time.time()
            try:
                result = test_func()
                duration = time.time() - start_time
                results.append((test_name, result, duration))
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results.append((test_name, False, 0))
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 30)
        passed = 0
        for test_name, result, duration in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status} ({duration:.2f}s)")
            if result:
                passed += 1
        
        print(f"\nTotal: {passed}/{len(results)} tests passed")
        
        if passed == len(results):
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️ Some tests failed. Check the API server.")

def main():
    """Main function"""
    import sys
    
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🎯 Testing API at: {base_url}")
    
    tester = APITester(base_url)
    tester.test_all()

if __name__ == "__main__":
    main()