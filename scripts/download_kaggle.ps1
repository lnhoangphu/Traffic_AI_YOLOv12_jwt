<#
Yêu cầu:
- pip install kaggle
- Đặt kaggle.json ở thư mục gốc repo
#>

$ErrorActionPreference = "Stop"

# Thư mục chứa kaggle.json (repo root)
$repoRoot = (Get-Location).Path
$kaggleJson = Join-Path $repoRoot "kaggle.json"
if (-not (Test-Path $kaggleJson)) {
  Write-Error "Không tìm thấy kaggle.json ở $repoRoot. Hãy đặt file kaggle.json tại thư mục gốc repo."
}

# Chỉ đạo Kaggle CLI dùng kaggle.json ở repo
$env:KAGGLE_CONFIG_DIR = $repoRoot

# Thư mục đích
$dstRoot = Join-Path $repoRoot "datasets_src"
New-Item -ItemType Directory -Force -Path $dstRoot | Out-Null

# Danh sách dataset cần tải
$datasets = @(
  @{ slug = "programmerrdai/road-issues-detection-dataset"; folder = "road_issues" },
  @{ slug = "samuelayman/object-detection";                folder = "object_detection_35" },
  @{ slug = "hoangper007/vietnamses-traffic-sign-detection-augmentaion"; folder = "vn_traffic_sign" },
  @{ slug = "starsw/intersection-flow-5k";                 folder = "intersection_flow_5k" }
)


foreach ($d in $datasets) {
  $outDir = Join-Path $dstRoot $d.folder
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  Write-Host "Downloading $($d.slug) -> $outDir"
  kaggle datasets download -d $d.slug -p $outDir --unzip
}

Write-Host "Done. Datasets saved in $dstRoot"