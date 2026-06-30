import argparse
import json
import os
from datetime import datetime

import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(REPO_ROOT, "playground", "data", "eval")
OUTPUT_COLUMNS = [
    "model",
    "vqav2_3k",
    "gqa",
    "vizwiz",
    "sqa",
    "textvqa",
    "pope_accuracy",
    "mmbench",
    "mmbench_cn",
    "flickr_Average",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
    )
    return parser.parse_args()


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_one(model):
    paths = {
        "flickr": os.path.join(BASE_DIR, "flickr", "answers", model, "Result.json"),
        "sqa": os.path.join(BASE_DIR, "scienceqa", "answers", model, "Result.json"),
        "pope": os.path.join(BASE_DIR, "pope", "answers", model, "Result.json"),
        "textvqa": os.path.join(BASE_DIR, "textvqa", "answers", model, "Result.json"),
        "gqa": os.path.join(BASE_DIR, "gqa", "answers", "llava_gqa_testdev_balanced", model, "Result.json"),
        "vizwiz": os.path.join(BASE_DIR, "vizwiz", "answers", model, "Result.json"),
        "mmbench": os.path.join(BASE_DIR, "mmbench", "answers", "mmbench_dev_20230712", model, "Result.json"),
        "mmbench_cn": os.path.join(BASE_DIR, "mmbench", "answers", "mmbench_dev_cn_20231003", model, "Result.json"),
        "vqav2_3k": os.path.join(BASE_DIR, "vqav2", "answers", "vqav2_test_3k", model, "Result.json"),
    }

    flickr = read_json(paths["flickr"])
    pope = read_json(paths["pope"])

    row = {
        "model": model,
        "vqav2_3k": as_float(read_json(paths["vqav2_3k"]).get("accuracy")),
        "gqa": as_float(read_json(paths["gqa"]).get("accuracy")),
        "vizwiz": as_float(read_json(paths["vizwiz"]).get("accuracy")),
        "sqa": as_float(read_json(paths["sqa"]).get("acc") or read_json(paths["sqa"]).get("accuracy")),
        "textvqa": as_float(read_json(paths["textvqa"]).get("accuracy")),
        "pope_accuracy": as_float(pope.get("accuracy")),
        "mmbench": as_float(read_json(paths["mmbench"]).get("accuracy")),
        "mmbench_cn": as_float(read_json(paths["mmbench_cn"]).get("accuracy")),
        "flickr_Average": as_float(flickr.get("Average")),
    }
    return row


def main():
    args = get_args()
    rows = [collect_one(model) for model in args.models]
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    output_file = args.output_file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(BASE_DIR, "summary", f"eval_summary_{timestamp}.xlsx")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        df.to_excel(output_file, index=False)
    except ModuleNotFoundError:
        output_file = os.path.splitext(output_file)[0] + ".csv"
        df.to_csv(output_file, index=False)
    print(f"Saved summary to {output_file}")


if __name__ == "__main__":
    main()
