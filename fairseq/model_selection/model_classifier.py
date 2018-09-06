#! /usr/bin/python
# -*- coding: utf-8 -*-

"""The model classifier.

Training:
    Dev data (x, y), eval on M models, get best BLEU on model m: classification task [(x, y) -> m]
Prediction:
    Test data x -> model index m
"""

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .. import tasks, options, utils
from ..data import IndexedRawTextDataset, data_utils
from ..models.transformer import Embedding, PositionalEmbedding, base_architecture
from . import utils as ms_utils

__author__ = 'fyabc'


class ClassifierDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
        assert len(x_list) == len(y_list)
        self.sizes = np.array([len(x) for x in self.x_list])

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, index):
        return {
            'id': index,
            'x': self.x_list[index],
            'y': self.y_list[index],
        }

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def make_collater(src_dict, left_pad=True):
        pad_idx = src_dict.pad()
        eos_idx = src_dict.eos()

        def collater(samples):
            if len(samples) == 0:
                return {}

            def merge(key, left_pad, move_eos_to_beginning=False):
                return data_utils.collate_tokens(
                    [s[key] for s in samples],
                    pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                )

            id_ = torch.LongTensor([s['id'] for s in samples])
            src_tokens = merge('x', left_pad=left_pad)
            # sort by descending source length
            src_lengths = torch.LongTensor([s['x'].numel() for s in samples])
            src_lengths, sort_order = src_lengths.sort(descending=True)
            id_ = id_.index_select(0, sort_order)
            src_tokens = src_tokens.index_select(0, sort_order)

            target = torch.LongTensor([s['y'] for s in samples])
            target = target.index_select(0, sort_order)

            batch = {
                'id': id_,
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
                'target': target,
            }

            return batch

        return collater

    def make_dataloader(self, args, src_dict, shuffle: bool, batch_size: int=32, left_pad: bool=True):
        if shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        indices = np.argsort(self.sizes[indices], kind='mergesort')

        indices = data_utils.filter_by_size(
            indices, self.size, max_positions=args.max_source_positions, raise_exception=True)
        batch_sampler = data_utils.batch_by_size(
            indices, self.size, max_tokens=None, max_sentences=batch_size,
            required_batch_size_multiple=1,
        )
        return DataLoader(self, batch_sampler=batch_sampler, collate_fn=self.make_collater(src_dict, left_pad))


class ModelClassifier(nn.Module):
    def __init__(self, args, task, left_pad=True, model_state=None):
        super().__init__()

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        self.embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
        embed_dim = self.embed_tokens.embedding_dim
        self.padding_idx = self.embed_tokens.padding_idx

        self.embed_scale = math.sqrt(embed_dim)

        no_token_positional_embeddings = False
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=False,     # [NOTE]: In LSTM, always converted to right pad
            learned=args.encoder_learned_pos,
        ) if not no_token_positional_embeddings else None

        if model_state is not None:
            self.embed_tokens.weight.data = model_state['encoder.embed_tokens.weight']
            self.embed_positions._float_tensor.data = model_state['encoder.embed_positions._float_tensor']

        self.left_pad = left_pad
        self.padding_value = 0.
        self.hidden_size = 32
        self.num_layers = 2
        self.bidirectional = True
        self.dropout_out = 0.0
        self.output_units = self.hidden_size
        if self.bidirectional:
            self.output_units *= 2
        self.n_classes = len(ms_utils.CheckPoints)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_out if self.num_layers > 1 else 0.,
            bidirectional=self.bidirectional,
        )

        self.linear = nn.Linear(self.output_units, self.n_classes)

    def forward(self, src_tokens, src_lengths):
        """

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            1-D LongTensor. Indicate which model to select for each sentence in the batch.
        """

        bsz, seqlen = src_tokens.size()

        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        # if self.bidirectional:
        #     state_size = 2 * self.num_layers, bsz, self.hidden_size
        # else:
        #     state_size = self.num_layers, bsz, self.hidden_size
        # h0 = x.data.new(*state_size).zero_()
        # c0 = x.data.new(*state_size).zero_()
        packed_outs, _ = self.lstm(packed_x)

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        padding_mask = src_tokens.eq(self.padding_idx)

        # Output linear
        print(padding_mask)


def read_classifier_data(args, task: tasks.FairseqTask, test_size=0.1):
    src_dict = task.source_dictionary
    x_list = IndexedRawTextDataset(ms_utils.TargetFiles['x'], src_dict).tokens_list
    with open(ms_utils.TargetFiles['y'], 'r', encoding='utf-8') as f_y:
        y_list = [int(line.strip()) for line in f_y]
    split = int(len(x_list) * test_size)
    train_dataset = ClassifierDataset(x_list[split:], y_list[split:])
    test_dataset = ClassifierDataset(x_list[:split], y_list[:split])
    train_dataloader = train_dataset.make_dataloader(args, src_dict, shuffle=True, batch_size=32)
    test_dataloader = test_dataset.make_dataloader(args, src_dict, shuffle=False, batch_size=32)

    return {
        'train': train_dataloader,
        'test': test_dataloader,
    }


def main():
    # Get training args.
    old_sys_argv = sys.argv
    try:
        sys.argv = [
            'train.py', ms_utils.DataDir,
            '-a', 'transformer_vaswani_wmt_en_de_big',
            '--share-all-embeddings',
            '--optimizer', 'adam',
            '--adam-betas', '(0.9, 0.98)',
            '--clip-norm', '0.0',
            '--lr-scheduler', 'inverse_sqrt',
            '--warmup-init-lr', '1e-07',
            '--warmup-updates', '4000',
            '--lr', '0.001',
            '--min-lr', '1e-09',
            '--update-freq', '16',
            '--dropout', '0.3',
            '--weight-decay', '0.0',
            '--criterion', 'label_smoothed_cross_entropy',
            '--label-smoothing', '0.1',
            '--max-tokens', '4096',
            '--no-progress-bar',
            '--save-dir', '.',
        ]
        parser = options.get_training_parser()
        args = options.parse_args_and_arch(parser)
    finally:
        sys.argv = old_sys_argv

    # Build model.
    task = tasks.setup_task(args)
    model_state = ms_utils.get_model_state(ms_utils.average_checkpoints())
    classifier = ModelClassifier(args, task, model_state=model_state)

    # Read training data.
    data = read_classifier_data(args, task)

    # Training.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters())

    step = 0
    for epoch in range(1, 100 + 1):
        for sample in data['train']:
            model_output = classifier(**sample['net_input'])
            step += 1
            if step == 5:
                exit()
