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
NAME=$1
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_BASE_DIR}/${NAME} \
    --question-file ${REPO_ROOT}/playground/data/eval/okvqa/llava_okvqa_val.jsonl \
    --image-folder "${OKVQA_IMAGE_FOLDER}" \
    --answers-file ${REPO_ROOT}/playground/data/eval/okvqa/answers/$NAME/${CHUNKS}_${IDX}.jsonl \
    --temperature 0 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &

done

wait

output_file=${REPO_ROOT}/playground/data/eval/okvqa/answers/$NAME/merge.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${REPO_ROOT}/playground/data/eval/okvqa/answers/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_okvqa --annotation-file ${REPO_ROOT}/playground/data/eval/okvqa/okvqa_val.json --result-file ${output_file}
