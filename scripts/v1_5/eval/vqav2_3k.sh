#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
set -euo pipefail

CONFIG_FILE="${SCRIPT_DIR}/config.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found!"
    exit 1
fi

source "$CONFIG_FILE"

if [ -z "${MODEL_BASE_DIR+x}" ] || [ -z "${GPU_LIST+x}" ]; then
    echo "Error: MODEL_BASE_DIR or GPU_LIST not defined in $CONFIG_FILE"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

if [ "$CHUNKS" -eq 0 ]; then
    echo "Error: GPU_LIST is empty in $CONFIG_FILE"
    exit 1
fi

CKPT=$1
SPLIT="vqav2_test_3k"
DATA_DIR="${REPO_ROOT}/playground/data/eval/vqav2"
ANNOTATION_FILE="${DATA_DIR}/test_3k.json"
QUESTION_FILE="${DATA_DIR}/test_3k.jsonl"
IMAGE_FOLDER="${OKVQA_IMAGE_FOLDER}"
ANSWER_DIR="${DATA_DIR}/answers/${SPLIT}/${CKPT}"
OUTPUT_FILE="${ANSWER_DIR}/merge.jsonl"

if [ ! -f "$QUESTION_FILE" ]; then
    echo "Error: $QUESTION_FILE not found!"
    echo "Run: python ${REPO_ROOT}/scripts/convert_vqav2_3k_to_jsonl.py --input-file $ANNOTATION_FILE --output-file $QUESTION_FILE"
    exit 1
fi

pids=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "${MODEL_BASE_DIR}/${CKPT}" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if [ "$status" -ne 0 ]; then
    echo "Error: one or more evaluation workers failed."
    exit "$status"
fi

mkdir -p "$ANSWER_DIR"
> "$OUTPUT_FILE"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$OUTPUT_FILE"
done

python -m llava.eval.eval_vqav2_3k \
    --annotation-file "$ANNOTATION_FILE" \
    --result-file "$OUTPUT_FILE" \
    --output-dir "$ANSWER_DIR"
