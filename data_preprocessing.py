'''
Data Preprocessing: convert RRC dataset to model inputs.
Last update: Qinghe Li, 2020.01.27
'''

import torch
import json
from transformers import BertTokenizer


def get_start_and_end(context, start, text):
    if start - 1 < 0:
        start = 0
    else:
        start = start - 1

    start = context.find(text, start)
    end = start + len(text) - 1

    char2id = dict()
    i = 0
    for idx, c in enumerate(context):
        if c == ' ':
            i += 1
        else:
            char2id[idx] = i

    return char2id[start], char2id[end]


def extract_example(file_path, tokenizer, max_seq_length):
    all_input_ids = []
    all_masks = []
    all_segment_ids = []
    all_answer_starts = []
    all_answer_ends = []

    with open(file_path, 'r', encoding='utf-8') as reader:
        dataset = json.load(reader)['data']

        for line in dataset:
            paragraph = line['paragraphs'][0]
            tokens = tokenizer.tokenize(paragraph['context'])
            context = ' '.join(tokens)

            for qa in paragraph['qas']:
                sequence = tokens.copy()
                sequence.insert(0, '[CLS]')
                sequence.append('[SEP]')

                segment_ids = [0] * len(sequence)
                sequence.extend(tokenizer.tokenize(qa['question']))
                sequence.append('[SEP]')

                mask = [1] * len(sequence)
                mask.extend([0] * (max_seq_length - len(mask)))

                segment_ids.extend([1] * (max_seq_length - len(segment_ids)))

                start = qa['answers'][0]['answer_start']
                text = qa['answers'][0]['text']
                text = ' '.join(tokenizer.tokenize(text))

                answer_start, answer_end = get_start_and_end(context, start, text)
                input_ids = tokenizer.convert_tokens_to_ids(sequence)

                if len(input_ids) < max_seq_length:
                    input_ids.extend([0] * (max_seq_length - len(input_ids)))
                else:
                    input_ids = input_ids[: max_seq_length]

                all_input_ids.append(input_ids)
                all_masks.append(mask)
                all_segment_ids.append(segment_ids)
                all_answer_starts.append(answer_start)
                all_answer_ends.append(answer_end)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_masks = torch.tensor(all_masks, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    all_answer_starts = torch.tensor(all_answer_starts, dtype=torch.long)
    all_answer_ends = torch.tensor(all_answer_ends, dtype=torch.long)

    return all_input_ids, all_masks, all_segment_ids, all_answer_starts, all_answer_ends
