import os
import json
import argparse


"""
parse args: --split --ckpt
"""

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="llava_gqa_testdev_balanced")
parser.add_argument("--ckpt", type=str)
args = parser.parse_args()

CKPT = args.ckpt
SPLIT = args.split

src = f'./playground/data/gqa/answers/{SPLIT}/{CKPT}/merge.jsonl'
dst = f'/Data/haotian/gqa/testdev_balanced_predictions.json'

all_answers = []
for line_idx, line in enumerate(open(src)):
    res = json.loads(line)
    question_id = res['question_id']
    text = res['text'].rstrip('.').lower()
    all_answers.append({"questionId": question_id, "prediction": text})

with open(dst, 'w') as f:
    json.dump(all_answers, f)
