import os
import math
import argparse
import json
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
def coco_caption_eval(annotation_file, results_file):


    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    result = {}
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
        result[metric] = score * 100.

    metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
    selected = [result[x] for x in metrics if x in result]
    if selected:
        result["Average"] = sum(selected) / len(selected)
    if "Bleu_4" in result and "CIDEr" in result:
        result["Bleu_4_plus_CIDEr"] = result["Bleu_4"] + result["CIDEr"]

    return result

def main(args):
    results_file=args.result_file
    format_res_path=args.format_res_path
    with open(results_file, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    with open(format_res_path, "w") as f:
        json.dump(data, f, indent=2)
    result = coco_caption_eval(args.annotation_file, format_res_path)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "Result.json"), "w") as f:
            json.dump({k: f"{v:.2f}" for k, v in result.items()}, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--format-res-path', type=str)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    main(args)
