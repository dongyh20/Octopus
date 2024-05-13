import os
import argparse
import json

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    src = os.path.join('./playground/data/vizwiz/answers', args.split, f'{args.ckpt}.jsonl')
    test_split = f'./playground/data/vizwiz/llava_{args.split}.jsonl'
    dst = os.path.join('./playground/data/vizwiz/answers_upload', args.split, f'{args.ckpt}.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    results = []
    error_line = 0
    for line_idx, line in enumerate(open(src)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1
    results = {x['question_id']: x['text'] for x in results}
    test_split = [json.loads(line) for line in open(test_split)]
    split_ids = set([x['question_id'] for x in test_split])

    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []

    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        assert x['question_id'] in results
        all_answers.append({
            'image': x['image'],
            'answer': answer_processor(results[x['question_id']])
        })

    with open(dst, 'w') as f:
        json.dump(all_answers, open(dst, 'w'))
