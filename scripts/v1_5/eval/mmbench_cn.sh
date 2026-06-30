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

SPLIT="mmbench_dev_cn_20231003"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_mmbench \
    --model-path ${MODEL_BASE_DIR}/${NAME} \
    --question-file ${REPO_ROOT}/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ${REPO_ROOT}/playground/data/eval/mmbench/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
    --single-pred-prompt \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --lang cn &

done

wait

output_file=${REPO_ROOT}/playground/data/eval/mmbench/answers/$SPLIT/$NAME/merge.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${REPO_ROOT}/playground/data/eval/mmbench/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ${REPO_ROOT}/llava/eval/eval_mmbench.py \
    --annotation-file ${REPO_ROOT}/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-file $output_file \
    --output-dir ${REPO_ROOT}/playground/data/eval/mmbench/answers/$SPLIT/$NAME \
    --experiment $NAME



















