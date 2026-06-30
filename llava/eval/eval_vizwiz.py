import os
import argparse
import json
import re

from m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./playground/Instructions_slim/VizWiz/val_new.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_slim_new/VizWiz/Zero_shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str,default=None)
    return parser.parse_args()

def eval_single(annotation_file, result_file):
    annotations=[]
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = [json.loads(line) for line in f if line.strip()]
    annotations = {annotation['question_id']: annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    for result in results:
        annotation = annotations[result['question_id']]
        pred = result['text']
        pred = pred[1:] if len(pred) > 0 and pred[0] == ' ' else pred
        ground_truth = annotation['answer']
        if pred.upper() == ground_truth.upper():
            right += 1

    accuracy = 100. * right / total
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, accuracy))
    #将结果写入文件
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, 'Result.json')
        with open(output_file, 'w') as f:
            json.dump({'accuracy': f'{accuracy:.2f}', 'samples': total}, f, indent=2)
    



if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
