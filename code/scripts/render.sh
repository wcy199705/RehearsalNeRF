#!/bin/bash
# Render from a trained RehearsalNeRF checkpoint
#
# Usage:
#   bash code/scripts/render.sh <config_path> <log_dir> <model_name> <mode> [gpu_id]   (from the project root)
#
# Modes:
#   --render-only       Full scene rendering
#   --validate-only     Validation metrics
#   --hue_test          Hue variation rendering
#   --frozen_light      Fixed lighting, varying motion
#   --frozen_motion     Fixed motion, varying lighting
#   --intensity_left    Left intensity map
#   --intensity_right   Right intensity map
#   --intensity_center  Center intensity map
#   --intensity_square  Square intensity map
#   --intensity_lr      Left-right intensity map
#   --decomp_rendering_only  Decomposed rendering
#   --render_fig8       Figure-8 camera path
#
# Examples:
#   bash code/scripts/render.sh plenoxels/configs/scenes/01.py logs/KPOP/01 120000model.pth --render-only
#   bash code/scripts/render.sh plenoxels/configs/scenes/01.py logs/KPOP/01 120000model.pth --hue_test 2

CONFIG=${1:?Usage: bash scripts/render.sh <config> <log_dir> <model_name> <mode> [gpu_id]}
LOG_DIR=${2:?Missing log_dir}
MODEL=${3:?Missing model_name}
MODE=${4:?Missing mode (e.g. --render-only, --hue_test)}
GPU=${5:-0}

cd "$(dirname "$0")/../plenoxels" || exit 1

PYTHONPATH='.' CUDA_VISIBLE_DEVICES="$GPU" python ./plenoxels/main.py \
  --config-path "./$CONFIG" \
  --log-dir "$LOG_DIR" \
  --model_name "$MODEL" \
  "$MODE"
