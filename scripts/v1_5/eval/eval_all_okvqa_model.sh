#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
set -euo pipefail

LOG_FILE="${SCRIPT_DIR}/eval_all_okvqa_model.log"


exec > "$LOG_FILE" 2>&1

models=(
 "llava-v1.5-7b-okvqa-cgm"

)

for model in "${models[@]}"; do
    echo "========================================"
    echo "Evaluating model: $model"
    echo "========================================"
    
    
    bash "${SCRIPT_DIR}/okvqa.sh" "$model"
    bash "${SCRIPT_DIR}/sqa.sh" "$model"
    bash "${SCRIPT_DIR}/pope.sh" "$model"
    bash "${SCRIPT_DIR}/textvqa.sh" "$model"
    bash "${SCRIPT_DIR}/gqa.sh" "$model"
    bash "${SCRIPT_DIR}/vizwiz1.sh" "$model"
    bash "${SCRIPT_DIR}/mmbench.sh" "$model"
    bash "${SCRIPT_DIR}/mmbench_cn.sh" "$model"
    bash "${SCRIPT_DIR}/vqav2_3k.sh" "$model"
    
    
    if [ $? -ne 0 ]; then
        echo "Error occurred during evaluation of $model"
        exit 1
    fi

    python ${REPO_ROOT}/scripts/collect_eval_results.py \
        --models "$model" \
        --output-file ${REPO_ROOT}/playground/data/eval/summary/${model}_summary.xlsx
done

echo "All models evaluated successfully."
