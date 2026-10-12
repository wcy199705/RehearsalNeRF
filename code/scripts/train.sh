#!/bin/bash
# Train a RehearsalNeRF model
#
# Usage:
#   bash code/scripts/train.sh <config_path> [gpu_id]   (from the project root)
#
# Examples:
#   bash code/scripts/train.sh plenoxels/configs/scenes/01.py
#   bash code/scripts/train.sh plenoxels/configs/scenes/01.py 2

CONFIG=${1:?Usage: bash scripts/train.sh <config_path> [gpu_id]}
GPU=${2:-0}

cd "$(dirname "$0")/../plenoxels" || exit 1

PYTHONPATH='.' CUDA_VISIBLE_DEVICES="$GPU" python ./plenoxels/main.py \
  --config-path "./$CONFIG"
