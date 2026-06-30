import argparse
import copy
import json
import os
import re
from datetime import datetime

import pandas as pd


OPTION_CANDIDATES = ["A", "B", "C", "D", "E"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="mmbench")
    return parser.parse_args()


def is_none(value):
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.lower() in ("nan", "none"):
        return True
    return False


def load_options(row):
    return {
        key: str(row[key])
        for key in OPTION_CANDIDATES
        if key in row and not is_none(row[key])
    }


def can_infer_option(answer, choices):
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        return sum(1 for choice in choices if prefix + choice + suffix in splits)

    answer_mod = copy.copy(answer)
    for char in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(char, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for choice in choices:
            if choice in splits and splits.index(choice) > (len(splits) - 5):
                return choice
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(value)) for value in choices.values()):
        return False
    candidates = []
    for key, value in choices.items():
        if str(value).lower() in answer:
            candidates.append(key)
    if len(candidates) == 1:
        return candidates[0]
    return False


def extract_prediction(pred, options):
    pred = str(pred)
    option_pred = can_infer_option(pred, options)
    if option_pred:
        return option_pred
    text_pred = can_infer_text(pred, options)
    if text_pred:
        return text_pred
    return "Z"


def build_group_key(row):
    if "g_index" in row and not is_none(row["g_index"]):
        return int(row["g_index"])
    if "image" in row:
        return (row["image"], row.get("hint", None), row["question"])
    return row["index"]


def clean_string_for_excel(text):
    if pd.isna(text) or text is None:
        return text
    return re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", str(text))


def clean_dataframe_for_excel(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].apply(
                lambda x: clean_string_for_excel(x) if isinstance(x, str) else x
            )
    return df_clean


def eval_single(annotation_file, result_file, output_dir, experiment):
    annotation_df = pd.read_table(annotation_file)
    df = annotation_df.copy()
    annotations = {int(row["index"]): row for _, row in df.iterrows()}
    results = [json.loads(line) for line in open(result_file)]

    predictions = {}
    extracted_predictions = {}
    for result in results:
        question_id = int(result["question_id"])
        if question_id not in annotations:
            continue
        annotation = annotations[question_id]
        pred = extract_prediction(result["text"], load_options(annotation))
        predictions[question_id] = pred
        extracted_predictions[question_id] = pred

    df["prediction"] = [predictions.get(int(idx), "Z") for idx in df["index"]]
    df["hit"] = [
        pred == str(answer).strip().upper()
        for pred, answer in zip(df["prediction"], df["answer"])
    ]
    df["group_key"] = [build_group_key(row) for _, row in df.iterrows()]

    total = len(df)
    correct = int(df["hit"].sum())
    row_accuracy = 100.0 * correct / total if total else 0.0

    group_hits = df.groupby("group_key", dropna=False)["hit"].all()
    circular_total = len(group_hits)
    circular_correct = int(group_hits.sum())
    circular_accuracy = 100.0 * circular_correct / circular_total if circular_total else 0.0

    print(
        "Samples: {}\nRow Accuracy: {:.2f}%\nCircular Samples: {}\nCircular Accuracy: {:.2f}%\n".format(
            total, row_accuracy, circular_total, circular_accuracy
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "Result.json"), "w") as f:
        json.dump(
            {
                "accuracy": f"{circular_accuracy:.2f}",
                "metric": "circular_accuracy",
                "circular_samples": circular_total,
                "circular_correct": circular_correct,
                "row_accuracy": f"{row_accuracy:.2f}",
                "row_samples": total,
                "row_correct": correct,
            },
            f,
            indent=2,
        )

    cur_df = annotation_df.copy()
    drop_columns = ["hint", "category", "source", "image", "comment", "l2-category"]
    cur_df = cur_df.drop(columns=[col for col in drop_columns if col in cur_df.columns])
    insert_at = cur_df.columns.get_loc("answer") if "answer" in cur_df.columns else len(cur_df.columns)
    cur_df.insert(insert_at, "prediction", None)
    for question_id, pred in predictions.items():
        cur_df.loc[df["index"] == question_id, "prediction"] = pred

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"{experiment}_{timestamp}.xlsx")
    export_df = clean_dataframe_for_excel(cur_df)
    try:
        export_df.to_excel(output_file, index=False, engine="openpyxl")
    except ModuleNotFoundError:
        output_file = os.path.join(output_dir, f"{experiment}_{timestamp}.csv")
        export_df.to_csv(output_file, index=False)
    print("Results saved to {}".format(output_file))


if __name__ == "__main__":
    args = get_args()
    eval_single(args.annotation_file, args.result_file, args.output_dir, args.experiment)
