import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def normalize_answer(text):
    return str(text).rstrip(".").lower()


def main():
    args = get_args()
    questions = json.load(open(args.question_file))
    answers = [json.loads(line) for line in open(args.result_file)]
    predictions = {
        str(answer["question_id"]): normalize_answer(answer["text"])
        for answer in answers
    }

    total = 0
    correct = 0
    for question_id, question in questions.items():
        if not question.get("isBalanced", False):
            continue
        total += 1
        if predictions.get(str(question_id), "") == normalize_answer(question["answer"]):
            correct += 1

    accuracy = 100.0 * correct / total if total else 0.0
    print("Samples: {}\nAccuracy: {:.2f}%\n".format(total, accuracy))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "Result.json"), "w") as f:
        json.dump(
            {"accuracy": f"{accuracy:.2f}", "samples": total, "correct": correct},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
