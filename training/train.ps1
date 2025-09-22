# YOLOv12 Traffic AI Training Scripts
# Cập nhật cho dữ liệu đã tiền xử lý

$ErrorActionPreference = "Stop"

Write-Host "🚀 YOLOv12 Traffic AI Training" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Kiểm tra dữ liệu đã tiền xử lý
$BALANCED_DATA = "datasets/traffic_ai_balanced_11class_processed/data.yaml"
$IMBALANCED_DATA = "datasets/traffic_ai_imbalanced_11class_processed/data.yaml"
$MODEL_WEIGHTS = "yolo12n.pt"

# Kiểm tra files cần thiết
if (-not (Test-Path $MODEL_WEIGHTS)) {
    Write-Error "❌ Không tìm thấy model weights: $MODEL_WEIGHTS"
    Write-Host "Hãy chạy: python scripts/download_yolo12n.py" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $BALANCED_DATA)) {
    Write-Error "❌ Không tìm thấy balanced data: $BALANCED_DATA"
    Write-Host "Hãy chạy tiền xử lý dữ liệu trước" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $IMBALANCED_DATA)) {
    Write-Error "❌ Không tìm thấy imbalanced data: $IMBALANCED_DATA"
    Write-Host "Hãy chạy tiền xử lý dữ liệu trước" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Tất cả files cần thiết đã sẵn sàng" -ForegroundColor Green

Write-Host "`n📋 TRAINING OPTIONS:" -ForegroundColor Cyan
Write-Host "1. Train Balanced Dataset" -ForegroundColor White
Write-Host "2. Train Imbalanced Dataset" -ForegroundColor White
Write-Host "3. Train Both Sequentially" -ForegroundColor White
Write-Host "4. Compare Existing Results" -ForegroundColor White
Write-Host "5. Full Pipeline (Train Both + Compare)" -ForegroundColor White
Write-Host "6. Interactive Master Script" -ForegroundColor White

$choice = Read-Host "`nChọn option (1-6)"

switch ($choice) {
    "1" {
        Write-Host "`n🎯 Training Balanced Dataset..." -ForegroundColor Yellow
        python training/train_balanced.py
    }
    
    "2" {
        Write-Host "`n🎯 Training Imbalanced Dataset..." -ForegroundColor Yellow
        python training/train_imbalanced.py
    }
    
    "3" {
        Write-Host "`n🔬 Training Both Datasets..." -ForegroundColor Yellow
        python training/train_balanced.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n⏳ Starting imbalanced training..." -ForegroundColor Cyan
            python training/train_imbalanced.py
        } else {
            Write-Error "❌ Balanced training failed. Stopping."
        }
    }
    
    "4" {
        Write-Host "`n📊 Comparing Existing Results..." -ForegroundColor Yellow
        python training/compare_results.py
    }
    
    "5" {
        Write-Host "`n🎯 Full Training Pipeline..." -ForegroundColor Yellow
        # Run balanced
        python training/train_balanced.py
        if ($LASTEXITCODE -eq 0) {
            # Run imbalanced
            python training/train_imbalanced.py
            if ($LASTEXITCODE -eq 0) {
                # Compare results
                python training/compare_results.py
            }
        }
    }
    
    "6" {
        Write-Host "`n� Interactive Master Script..." -ForegroundColor Yellow
        python training/run_training_pipeline.py
    }
    
    default {
        Write-Error "❌ Invalid choice: $choice"
        exit 1
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n🎉 OPERATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "📁 Check runs/ directory for results" -ForegroundColor Cyan
    Write-Host "📊 Use compare_results.py for detailed analysis" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ OPERATION FAILED!" -ForegroundColor Red
    Write-Host "Check error messages above" -ForegroundColor Yellow
}