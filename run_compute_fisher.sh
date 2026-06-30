#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/path/to/llava-model}"
PRE_MODEL_PATH="${PRE_MODEL_PATH:-/path/to/llava-1-5}"
DATA_PATH="${DATA_PATH:-/path/to/train_llava.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/images}"
FISHER_SAVE_PATH="${FISHER_SAVE_PATH:-/path/to/fisher_matrix.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/fisher-compute-output}"

python "${SCRIPT_DIR}/compute_fisher.py" \
 --model_name_or_path "${MODEL_PATH}" \
 --pre_model_name_or_path "${PRE_MODEL_PATH}" \
 --data_path "${DATA_PATH}" \
 --image_folder "${IMAGE_FOLDER}" \
 --fisher_save_path "${FISHER_SAVE_PATH}" \
 --output_dir "${OUTPUT_DIR}"
