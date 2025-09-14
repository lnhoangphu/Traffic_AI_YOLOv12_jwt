#!/usr/bin/env bash
set -euo pipefail

# Yêu cầu: pip install kaggle và có file ./kaggle.json
REPO_ROOT="$(pwd)"
if [ ! -f "$REPO_ROOT/kaggle.json" ]; then
  echo "Không tìm thấy kaggle.json ở $REPO_ROOT. Hãy đặt file kaggle.json tại thư mục gốc repo."
  exit 1
fi
export KAGGLE_CONFIG_DIR="$REPO_ROOT"

DST_ROOT="$REPO_ROOT/datasets_src"
mkdir -p "$DST_ROOT"

declare -A DATASETS
DATASETS[road_issues]="programmerrdai/road-issues-detection-dataset"
DATASETS[object_detection_35]="samuelayman/object-detection"
DATASETS[vn_traffic_sign]="hoangper007/vietnamses-traffic-sign-detection-augmentaion"
DATASETS[intersection_flow_5k]="starsw/intersection-flow-5k"

for folder in "${!DATASETS[@]}"; do
  slug="${DATASETS[$folder]}"
  out="$DST_ROOT/$folder"
  mkdir -p "$out"
  echo "Downloading $slug -> $out"
  kaggle datasets download -d "$slug" -p "$out" --unzip
done

echo "Done. Datasets saved in $DST_ROOT"