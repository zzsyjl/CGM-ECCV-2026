import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    return parser.parse_args()


def normalize_image_path(image_path):
    for prefix in ("COCO2014/val2014/", "val2014/"):
        if image_path.startswith(prefix):
            return image_path[len(prefix):]
    return image_path


def main():
    args = get_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    with open(args.output_file, "w") as f:
        for item in data:
            row = dict(item)
            row["image"] = normalize_image_path(row["image"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
