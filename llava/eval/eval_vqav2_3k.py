import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def eval_single(annotation_file, result_file, output_dir=None):
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    annotations = {str(annotation["question_id"]): annotation for annotation in annotations}

    with open(result_file, "r") as f:
        results = [json.loads(line) for line in f]

    total = len(results)
    right = 0
    for result in results:
        annotation = annotations[str(result["question_id"])]
        pred = result["text"]
        ground_truth = annotation["answer"]
        if pred.upper() == ground_truth.upper():
            right += 1

    accuracy = 100.0 * right / total if total else 0.0
    print("Samples: {}\nAccuracy: {:.2f}%\n".format(total, accuracy))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "Result.json")
        with open(output_file, "w") as f:
            json.dump({"accuracy": f"{accuracy:.2f}"}, f, indent=2)


if __name__ == "__main__":
    args = get_args()
    eval_single(args.annotation_file, args.result_file, args.output_dir)
