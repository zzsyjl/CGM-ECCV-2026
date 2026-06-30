#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}
CKPT=$1


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_caption_loader \
        --model-path ${MODEL_BASE_DIR}/$CKPT \
        --question-file ${REPO_ROOT}/playground/data/eval/flickr/flickr_test_questions.jsonl \
        --image-folder "${FLICKR_IMAGE_FOLDER}" \
        --answers-file ${REPO_ROOT}/playground/data/eval/flickr/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${REPO_ROOT}/playground/data/eval/flickr/answers/$CKPT/merge.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${REPO_ROOT}/playground/data/eval/flickr/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ${REPO_ROOT}/llava/eval/eval_caption.py \
    --annotation-file ${REPO_ROOT}/playground/data/eval/flickr/flickr_coco_annotation.json \
    --result-file ${output_file} \
    --format-res-path ${REPO_ROOT}/playground/data/eval/flickr/answers_converted/${CKPT}.json \
    --output-dir ${REPO_ROOT}/playground/data/eval/flickr/answers/$CKPT
