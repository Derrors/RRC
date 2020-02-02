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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data_preprocessing import extract_example


def to_list(tensor):
    '''
    Get the data from the GPUs and convert to lists.
    '''
    return tensor.detach().cpu().tolist()


def convert_data(starts, ends):
    '''
    Convert the answer positions to answer spans list.
    A list of numbers represent a list of words of the answers.
    '''
    assert len(starts) == len(ends)

    spans = []
    for start, end in zip(starts, ends):
        span = np.arange(start, end + 1)
        spans.append(span)
    return spans


def train(args, dataloader, model, optimizer):
    '''
    Training the BERT model on RRC dataset.
    '''

    loss_avg = 0                # the average loss during a complete epoch of training
    model.train()
    for batch_data in dataloader:
        batch_data = tuple(d.cuda() for d in batch_data)
        input_ids, masks, segment_ids, answer_starts, answer_ends = batch_data

        loss, _, _ = model(input_ids=input_ids,
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

        loss_avg += loss.item()
    loss_avg = loss_avg / len(dataloader)

    return loss_avg


def test(args, dataloader, model):
    '''
    Testing the BERT model on RRC dataset.
    '''
    ground_truth = []
    prediction = []

    model.eval()
    for batch_data in dataloader:
        batch_data = tuple(d.cuda() for d in batch_data)
        input_ids, masks, segment_ids, answer_starts, answer_ends = batch_data

        with torch.no_grad():
            start_scores, end_scores = model(input_ids=input_ids,
                                             attention_mask=masks,
                                             token_type_ids=segment_ids)

        # convert start and end scores to index
        pred_starts = np.argmax(to_list(start_scores), axis=1)
        pred_ends = np.argmax(to_list(end_scores), axis=1)
        pred_spans = convert_data(pred_starts, pred_ends)
        prediction.extend(pred_spans)
        true_spans = convert_data(to_list(answer_starts), to_list(answer_ends))
        ground_truth.extend(true_spans)

    assert len(prediction) == len(ground_truth)
    return prediction, ground_truth


def evaluate(prediction, ground_truth):
    '''
    Evaluate the trained model on the testing dataset.
    Return the EM(Exact Match) and F1 scores.
    '''
    def exact_match(pred, true):
        if list(pred) == list(true):
            return 1
        else:
            return 0

    def f1_score(pred, true):
        if list(pred) == list(true):
            return 1.0
        else:
            num_same = 0
            for n in pred:
                if n in true:
                    num_same += 1

            if len(pred) == 0:
                precision = 0.0
            else:
                precision = 1.0 * num_same / len(pred)
            recall = 1.0 * num_same / len(true)
            # avoid dividing zero
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
            return f1

    num_exact_match = 0
    f1 = 0
    for pred, true in zip(prediction, ground_truth):
        num_exact_match += exact_match(pred, true)
        f1 += f1_score(pred, true)

    metrics = {
        'EM': num_exact_match / len(prediction),
        'F1': f1 / len(prediction)
    }
    return metrics


def train_and_test(args, tokenizer, model, optimizer):
    '''
    Model training and testing on RRC dataset.
    '''
    # load the preprocessed data
    train_input_ids, train_masks, train_segment_ids, train_answer_starts, train_answer_ends = extract_example(os.path.join(args.data_dir, 'train.json'), tokenizer, args.max_seq_length)

    # build the data loader of training and testing datasets
    train_data = TensorDataset(train_input_ids, train_masks, train_segment_ids, train_answer_starts, train_answer_ends)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    test_input_ids, test_masks, test_segment_ids, test_answer_starts, test_answer_ends = extract_example(os.path.join(args.data_dir, 'test.json'), tokenizer, args.max_seq_length)
    test_data = TensorDataset(test_input_ids, test_masks, test_segment_ids, test_answer_starts, test_answer_ends)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    # start trianing and testing: after every epoch, do a evaluation of the model
    for i in range(args.epochs):
        loss = train(args, train_dataloader, model, optimizer)

        test_pred, test_ground = test(args, test_dataloader, model)
        test_metrics = evaluate(test_pred, test_ground)

        print('Epoch[{:<2d}/{:<2d}]: loss: {:.4f}  EM: {:.4f}  F1:{:.4f}'.format(
            i + 1, args.epochs, loss, test_metrics['EM'] * 100, test_metrics['F1'] * 100))


if __name__ == '__main__':
    # Running arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert_model',
                        default='./bert-pretrained-model/bert-base-uncased',
                        type=str)

    parser.add_argument('--data_dir',
                        default='./data/original data/rrc/laptop',
                        type=str,
                        help='The input data dir containing json files.')

    parser.add_argument('--output_dir',
                        default='./saved model checkpoints',
                        type=str,
                        help='The output directory where the model checkpoints will be written.',
                        )

    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='Total batch size for training.')

    parser.add_argument('--epochs',
                        default=40,
                        type=int,
                        help='Total number of training epochs to perform.')

    parser.add_argument('--max_seq_length',
                        default=512,
                        type=int,
                        help='The maximum total input sequence length after WordPiece tokenization. \n'
                             'Sequences longer than this will be truncated, and sequences shorter \n'
                             'than this will be padded.')

    parser.add_argument('--learning_rate',
                        default=3e-5,
                        type=float,
                        help='The initial learning rate for Adam.')

    parser.add_argument('--fp16',
                        default=True,
                        action='store_true',
                        help='Whether to use 16-bit float precision instead of 32-bit')

    args = parser.parse_args()

    torch.manual_seed(2020)

    # load tokenizer and BERT model
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

    train_and_test(args, tokenizer, model, optimizer)
