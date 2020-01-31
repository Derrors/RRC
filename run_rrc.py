'''
Experiments on the RRC dataset.
Last update: Qinghe Li, 2020.01.30
'''

import os
import torch
import random
import argparse
import numpy as np

from apex.optimizers import FP16_Optimizer, FusedAdam
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from data_preprocessing import extract_example
from tqdm import trange


def train(args, tokenizer, model, optimizer):
    os.path.join(args.data_dir, 'train.json')

    all_input_ids, all_masks, all_segment_ids, all_answer_starts, all_answer_ends = extract_example(os.path.join(args.data_dir, 'train.json'), tokenizer, args.max_seq_length)

    train_data = TensorDataset(all_input_ids, all_masks, all_segment_ids, all_answer_starts, all_answer_ends)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model.train()

    for t in trange(args.epochs):
        loss_avg = 0
        for batch_data in train_dataloader:
            batch_data = tuple(d.cuda() for d in batch_data)
            input_ids, masks, segment_ids, answer_starts, answer_ends = batch_data

            loss, start_scores, end_scores = model(input_ids=input_ids,
                                                   attention_mask=masks,
                                                   token_type_ids=segment_ids,
                                                   start_positions=answer_starts,
                                                   end_positions=answer_ends)

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_avg += loss

        t.set_postfix(loss='{:05.3f}'.format((loss_avg / len(train_dataloader)).item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert_model',
                        default='./bert-pretrained-model/bert-base-uncased',
                        type=str)

    parser.add_argument('--data_dir',
                        default='./data/original data/rrc/laptop',
                        type=str,
                        help='The input data dir containing json files.')

    parser.add_argument("--output_dir",
                        default='./saved model checkpoints',
                        type=str,
                        help="The output directory where the model checkpoints will be written.",
                        )

    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument('--fp16',
                        default=True,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    args = parser.parse_args()

    torch.manual_seed(2020)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertForQuestionAnswering.from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    model.cuda()

    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad is True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train(args, tokenizer, model, optimizer)
