#!/usr/bin/env python3
"""
Training Master Script
Script tổng hợp để chạy tất cả quá trình training và comparison
"""

import sys
import subprocess
from pathlib import Path
import time

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️ {description} interrupted by user")
        return False

def main():
    """Main training pipeline"""
    training_dir = Path(__file__).parent
    
    print("🎯 YOLOv12 Complete Training Pipeline")
    print("=" * 50)
    print("1. Train Balanced Dataset")
    print("2. Train Imbalanced Dataset") 
    print("3. Train Both Sequentially")
    print("4. Compare Existing Results")
    print("5. Full Pipeline (Train Both + Compare)")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        # Train balanced only
        script_path = training_dir / "train_balanced.py"
        run_script(script_path, "Training Balanced Dataset")
        
    elif choice == "2":
        # Train imbalanced only
        script_path = training_dir / "train_imbalanced.py"
        run_script(script_path, "Training Imbalanced Dataset")
        
    elif choice == "3":
        # Train both sequentially
        print("\n🔄 Training both datasets sequentially...")
        
        # Train balanced first
        balanced_script = training_dir / "train_balanced.py"
        if run_script(balanced_script, "Training Balanced Dataset"):
            print("\n⏳ Waiting 5 seconds before starting imbalanced training...")
            time.sleep(5)
            
            # Train imbalanced second
            imbalanced_script = training_dir / "train_imbalanced.py"
            run_script(imbalanced_script, "Training Imbalanced Dataset")
        
    elif choice == "4":
        # Compare existing results
        compare_script = training_dir / "compare_results.py"
        run_script(compare_script, "Comparing Training Results")
        
    elif choice == "5":
        # Full pipeline
        print("\n🎯 Running complete training pipeline...")
        
        # Train balanced
        balanced_script = training_dir / "train_balanced.py"
        balanced_success = run_script(balanced_script, "Training Balanced Dataset")
        
        if balanced_success:
            print("\n⏳ Waiting 5 seconds before starting imbalanced training...")
            time.sleep(5)
            
            # Train imbalanced
            imbalanced_script = training_dir / "train_imbalanced.py"
            imbalanced_success = run_script(imbalanced_script, "Training Imbalanced Dataset")
            
            if imbalanced_success:
                print("\n⏳ Waiting 5 seconds before comparison...")
                time.sleep(5)
                
                # Compare results
                compare_script = training_dir / "compare_results.py"
                run_script(compare_script, "Comparing Training Results")
        
    else:
        print("❌ Invalid choice!")
        sys.exit(1)
    
    print(f"\n🎉 Pipeline completed!")
    print(f"📁 Check runs/ directory for all training outputs")

if __name__ == "__main__":
    main()