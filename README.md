# Curvature Guided Mixing

Code release for CGM and CGM+ model merging experiments.

## Install

This repository follows the LLaVA environment definition from the official LLaVA `main` branch `pyproject.toml`.

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
```

For training or Fisher computation with the full LLaVA trainer dependencies:

```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

If `flash-attn` is not compatible with your CUDA/PyTorch build, keep the base install and run without FlashAttention.

## Structure

- `cgm/`: CGM/CGM+ arguments, data loading, Fisher computation, model saving, and merge operators.
- `compute_fisher.py`: computes Fisher diagonals.
- `merge_cgm.py`: merges a fine-tuned model with a pretrained model using `--method cgm` or `--method cgm_plus`.
- `run_cgm.sh`, `run_cgm_plus.sh`: entry scripts with the original hyperparameters.
- `scripts/v1_5/eval/`: evaluation scripts used by `eval_all_flickr_model.sh` and `eval_all_okvqa_model.sh`.
- `playground/data/eval/`: local directory for downloaded non-image evaluation annotations and question files.

## Prepare Eval Data

For evaluation annotation files, download them from the Hugging Face dataset repo before running evaluation:

```bash
pip install -U huggingface_hub

export HF_EVAL_DATA_REPO=gaifan0125/CGM-eval-data
mkdir -p playground/data/eval
hf download "$HF_EVAL_DATA_REPO" --repo-type dataset --local-dir playground/data/eval
```

After downloading, the retained evaluation scripts expect these files:

```text
playground/data/eval/flickr/flickr_coco_annotation.json
playground/data/eval/flickr/flickr_test_questions.jsonl
playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl
playground/data/eval/gqa/eval/eval.py
playground/data/eval/gqa/eval/testdev_balanced_questions.json
playground/data/eval/mmbench/mmbench_dev_20230712.tsv
playground/data/eval/mmbench/mmbench_dev_cn_20231003.tsv
playground/data/eval/okvqa/llava_okvqa_val.jsonl
playground/data/eval/okvqa/okvqa_val.json
playground/data/eval/pope/llava_pope_test.jsonl
playground/data/eval/pope/coco/*.json
playground/data/eval/scienceqa/llava_test_CQM-A.json
playground/data/eval/scienceqa/pid_splits.json
playground/data/eval/scienceqa/problems.json
playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl
playground/data/eval/textvqa/TextVQA_0.5.1_val.json
playground/data/eval/vizwiz/qa_val.jsonl
playground/data/eval/vqav2/test_3k.json
playground/data/eval/vqav2/test_3k.jsonl
```

Image files are not included. Download or mount the image folders locally, then edit `scripts/v1_5/eval/config.sh` or export the same variables:

```bash
export MODEL_BASE_DIR=/path/to/llava/checkpoints
export GPU_LIST=0,1,2,3
export FLICKR_IMAGE_FOLDER=/path/to/flickr30k/Images
export OKVQA_IMAGE_FOLDER=/path/to/OK-VQA/val2014
export SCIENCEQA_IMAGE_FOLDER=/path/to/ScienceQA/images/test
export TEXTVQA_IMAGE_FOLDER=/path/to/TextVQA/train_images
export GQA_IMAGE_FOLDER=/path/to/gqa/images
export VIZWIZ_IMAGE_FOLDER=/path/to/VizWiz/images
```

Useful upstream image/data download references:

- COCO train2017: http://images.cocodataset.org/zips/train2017.zip
- COCO val2014: http://images.cocodataset.org/zips/val2014.zip
- GQA images: https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
- OCR-VQA download script: https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing
- TextVQA train/val images: https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
- VisualGenome part1: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
- VisualGenome part2: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
- ScienceQA images: https://github.com/lupantech/ScienceQA
- VizWiz annotations/images: https://vizwiz.cs.colorado.edu/VizWiz_final/
- MMBench TSVs: https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv and https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv

For the retained evaluation scripts, map the downloaded images as follows:

- `FLICKR_IMAGE_FOLDER`: Flickr30K image folder.
- `OKVQA_IMAGE_FOLDER`: COCO `val2014`; reused by OK-VQA, POPE, and VQAv2-3K.
- `SCIENCEQA_IMAGE_FOLDER`: ScienceQA `images/test`.
- `TEXTVQA_IMAGE_FOLDER`: extracted TextVQA train/val image folder.
- `GQA_IMAGE_FOLDER`: extracted GQA image folder.
- `VIZWIZ_IMAGE_FOLDER`: VizWiz image folder matching `qa_val.jsonl`.
- MMBench images are embedded in the TSV files downloaded under `playground/data/eval/mmbench`.

## Run Merging

Set paths through environment variables, then run:

```bash
FT_MODEL_PATH=/path/to/finetuned-model \
PRE_MODEL_PATH=/path/to/pretrained-model \
FISHER_PATH=/path/to/finetuned-fisher.bin \
PRE_FISHER_PATH=/path/to/pretrained-fisher.bin \
OUTPUT_DIR=/path/to/output-model \
bash run_cgm_plus.sh
```

For CGM, use `bash run_cgm.sh` with the same variables.

## Compute Fisher

```bash
MODEL_PATH=/path/to/model \
PRE_MODEL_PATH=/path/to/pretrained-model \
DATA_PATH=/path/to/train_llava.json \
IMAGE_FOLDER=/path/to/images \
FISHER_SAVE_PATH=/path/to/fisher_matrix.bin \
bash run_compute_fisher.sh
```

## Evaluation

After preparing image folders and checkpoints:

```bash
bash scripts/v1_5/eval/eval_all_flickr_model.sh
bash scripts/v1_5/eval/eval_all_okvqa_model.sh
```

The scripts write answers and `Result.json` files under `playground/data/eval/<dataset>/answers/`.

## Upstream References

- LLaVA README: https://github.com/haotian-liu/LLaVA/blob/main/README.md
- LLaVA pyproject: https://github.com/haotian-liu/LLaVA/blob/main/pyproject.toml
- LLaVA evaluation guide: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md