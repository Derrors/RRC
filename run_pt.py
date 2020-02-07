'''
BERT Post-Training for Review Reading Comprehension.
Last update: Qinghe Li, 2020.02.04
'''

import os
import torch
import argparse
import numpy as np

from apex.optimizers import FP16_Optimizer, FusedAdam
from transformers.modeling_bert import BertModel, BertPreTrainingHeads, BertPreTrainedModel
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

BERT_PATH = './bert-pretrained-model'
OUTPUT_PATH = './bert-pretrained-model'
DK_PATH = './data/domain_corpus'
MRC_PATH = './data/squad'


class BertForPostTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, mode, input_ids, attention_mask=None, token_type_ids=None, masked_lm_labels=None, next_sentence_label=None, start_positions=None, end_positions=None):
        sequence_output, pooler_output = self.bert(input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids)
        if mode == 'review':
            prediction_scores, seq_relationship_scores = self.cls(sequence_output, pooler_output)
            if masked_lm_labels is not None and next_sentence_label is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

                loss_mlm = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                loss_nsp = loss_fct(seq_relationship_scores.view(-1, 2), next_sentence_label.view(-1))
                loss_dk = loss_mlm + loss_nsp

                return loss_dk
            else:
                return prediction_scores, seq_relationship_scores
        elif mode == 'squad':
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            if start_positions is not None and end_positions is not None:
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignore_index = start_logits.size(1)
                start_positions.clamp_(0, ignore_index)
                end_positions.clamp_(0, ignore_index)

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                loss_start = loss_fct(start_logits, start_positions)
                loss_end = loss_fct(end_logits, end_positions)
                loss_mrc = (loss_start + loss_end) / 2

                return loss_mrc
            else:
                return start_logits, end_logits


def train(args, model, optimizer):
    review_examples = np.load(os.path.join(args.review_data, 'data.npz'))
    squad_examples = np.load(args.squad_data)

    review_train_data = TensorDataset(
        torch.from_numpy(review_examples['input_ids']),
        torch.from_numpy(review_examples['segment_ids']),
        torch.from_numpy(review_examples['input_mask']),
        torch.from_numpy(review_examples['masked_lm_ids']),
        torch.from_numpy(review_examples['next_sentence_labels'])
    )

    squad_train_data = TensorDataset(
        torch.from_numpy(squad_examples['input_ids']),
        torch.from_numpy(squad_examples['segment_ids']),
        torch.from_numpy(squad_examples['input_mask']),
        torch.from_numpy(squad_examples['start_positions']),
        torch.from_numpy(squad_examples['end_positions'])
    )

    review_dataloader = DataLoader(review_train_data, sampler=RandomSampler(review_train_data), batch_size=args.batch_size, drop_last=True)
    squad_dataloader = DataLoader(squad_train_data, sampler=RandomSampler(squad_train_data), batch_size=args.batch_size, drop_last=True)

    model.train()
    model.zero_grad()

    review_iter = iter(review_dataloader)
    squad_iter = iter(squad_dataloader)

    tag = True
    global_steps = 0

    if args.mode == 'DK':
        while tag:
            try:
                batch_review = next(review_iter)
            except BaseException:
                review_iter = iter(review_dataloader)
                batch_review = next(review_iter)
            batch_review = tuple(d.cuda() for d in batch_review)
            input_ids, segment_ids, input_mask, masked_lm_ids, next_sentence_labels = batch_review

            loss = model(mode='review',
                         input_ids=input_ids.long(),
                         attention_mask=input_mask.long(),
                         token_type_ids=segment_ids.long(),
                         masked_lm_labels=masked_lm_ids.long(),
                         next_sentence_label=next_sentence_labels.long())

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_steps += 1

            if global_steps % 1000 == 0:
                print('Step[{:>5d}/{:<5d}]: loss: {:.4f}'.format(global_steps, args.steps, loss.item()))
                model.float()
                torch.save(model.state_dict(), os.path.join(args.output_path, 'pytorch_model.bin'))
            if global_steps > 50000:
                tag = False
                break

    elif args.mode == 'MRC':
        while tag:
            try:
                batch_squad = next(squad_iter)
            except BaseException:
                squad_iter = iter(squad_dataloader)
                batch_squad = next(squad_iter)
            batch_squad = tuple(d.cuda() for d in batch_squad)
            input_ids, segment_ids, input_mask, start_positions, end_positions = batch_squad

            loss = model(mode='squad',
                         input_ids=input_ids.long(),
                         attention_mask=input_mask.long(),
                         token_type_ids=segment_ids.long(),
                         start_positions=start_positions.long(),
                         end_positions=end_positions.long())

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_steps += 1

            if global_steps % 1000 == 0:
                print('Step[{:>5d}/{:<5d}]: loss: {:.4f}'.format(global_steps, args.steps, loss.item()))
                model.float()
                torch.save(model.state_dict(), os.path.join(args.output_path, 'pytorch_model.bin'))
            if global_steps > 50000:
                tag = False
                break

    elif args.mode == 'PT':
        while tag:
            try:
                batch_review = next(review_iter)
            except BaseException:
                review_iter = iter(review_dataloader)
                batch_review = next(review_iter)
            batch_review = tuple(d.cuda() for d in batch_review)
            input_ids, segment_ids, input_mask, masked_lm_ids, next_sentence_labels = batch_review

            loss_dk = model(mode='review',
                            input_ids=input_ids.long(),
                            attention_mask=input_mask.long(),
                            token_type_ids=segment_ids.long(),
                            masked_lm_labels=masked_lm_ids.long(),
                            next_sentence_label=next_sentence_labels.long())
            try:
                batch_squad = next(squad_iter)
            except BaseException:
                squad_iter = iter(squad_dataloader)
                batch_squad = next(squad_iter)
            batch_squad = tuple(d.cuda() for d in batch_squad)
            input_ids, segment_ids, input_mask, start_positions, end_positions = batch_squad

            loss_mrc = model(mode='squad',
                             input_ids=input_ids.long(),
                             attention_mask=input_mask.long(),
                             token_type_ids=segment_ids.long(),
                             start_positions=start_positions.long(),
                             end_positions=end_positions.long())

            loss = loss_dk + loss_mrc

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_steps += 1

            if global_steps % 1000 == 0:
                print('Step[{:>5d}/{:<5d}]: loss: {:.4f}'.format(global_steps, args.steps, loss.item()))
                model.float()
                torch.save(model.state_dict(), os.path.join(args.output_path, 'pytorch_model.bin'))
            if global_steps > 50000:
                tag = False
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert_model',
                        default='bert-base-uncased',
                        type=str)

    parser.add_argument('--review_data',
                        default='laptop',
                        type=str,
                        help='The domain knowledge data files.')

    parser.add_argument('--squad_data',
                        default='data.npz',
                        type=str,
                        help='The MRC task-awareness data files.')

    parser.add_argument('--mode',
                        default='PT',
                        type=str,
                        help='The training mode for Posting-Training.')

    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='Total batch size for training.')

    parser.add_argument('--steps',
                        default=50000,
                        type=int,
                        help='Total number of training steps to perform.')

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
                        default=False,
                        action='store_true',
                        help='Whether to use 16-bit float precision instead of 32-bit')

    args = parser.parse_args()

    if args.mode == 'DK':
        args.output_path = os.path.join(OUTPUT_PATH, 'DK', args.review_data)
    elif args.mode == 'MRC':
        args.output_path = os.path.join(OUTPUT_PATH, 'MRC')
    elif args.mode == 'PT':
        args.output_path = os.path.join(OUTPUT_PATH, 'PT', args.review_data)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.bert_model = os.path.join(BERT_PATH, args.bert_model)
    args.review_data = os.path.join(DK_PATH, args.review_data)
    args.squad_data = os.path.join(MRC_PATH, args.squad_data)

    torch.manual_seed(2020)

    model = BertForPostTraining.from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    model.cuda()

    # Prepare optimizer
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

    train(args, model, optimizer)
