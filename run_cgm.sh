#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FT_MODEL_PATH="${FT_MODEL_PATH:-/path/to/llava-v1.5-7b-flickr30}"
PRE_MODEL_PATH="${PRE_MODEL_PATH:-/path/to/llava-1-5}"
FISHER_PATH="${FISHER_PATH:-/path/to/fisher_matrix_flickr_from_pretrain_full.bin}"
PRE_FISHER_PATH="${PRE_FISHER_PATH:-/path/to/fisher_matrix_pretrain_full.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/llava-v1.5-7b-flickr-cgm0}"

python "${SCRIPT_DIR}/merge_cgm.py" \
 --model_name_or_path "${FT_MODEL_PATH}" \
 --pre_model_name_or_path "${PRE_MODEL_PATH}" \
 --sparse_level 0.95 \
 --alpha 0.2 \
 --output_dir "${OUTPUT_DIR}" \
 --full_layers True \
 --method cgm \
 --fisher_path "${FISHER_PATH}" \
 --pre_fisher_path "${PRE_FISHER_PATH}"
