#! /usr/bin/python
# -*- coding: utf-8 -*-

import collections
import os

import torch

from ..my_utils import ProjectDir

__author__ = 'fyabc'

CheckPoints = [43, 58, 73, 85, 88]
Subsets = ['valid', 'test']

SourceDir = os.path.join(ProjectDir, 'data')
TargetFiles = {
    'x': os.path.join(ProjectDir, 'data', 'classifier-inputs.txt'),
    'y': os.path.join(ProjectDir, 'data', 'classifier-targets.txt'),
}

ModelDir = '/home/v-yaf/DataTransfer/fairseq/WMT14_EN_DE_CYCLE_4.5k'
DataDir = '/home/v-yaf/DataTransfer/fairseq/wmt14_en_de_joined_dict'


def get_data_path(subset, lang):
    return os.path.join(DataDir, '{}.en-de.{}'.format(subset, lang))


def get_dict_path(lang):
    return os.path.join(DataDir, 'dict.{}.txt'.format(lang))


def get_bleu_path(subset, ckpt):
    return os.path.join(ProjectDir, 'data', 'bleu-{}-checkpoint{}.pt.txt'.format(subset, ckpt))


def get_model_path(ckpt):
    return os.path.join(ModelDir, 'checkpoint{}.pt'.format(ckpt))


def average_checkpoints(ckpt_list: list = None):
    if ckpt_list is None:
        ckpt_list = [get_model_path(ckpt) for ckpt in CheckPoints]

    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    for f in ckpt_list:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            if k not in params_dict:
                params_dict[k] = []
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            params_dict[k].append(p)

    averaged_params = collections.OrderedDict()
    # v should be a list of torch Tensor.
    for k, v in params_dict.items():
        summed_v = None
        for x in v:
            summed_v = summed_v + x if summed_v is not None else x
        averaged_params[k] = summed_v / len(v)
    new_state['model'] = averaged_params
    return new_state


def get_embedding(new_state):
    print(new_state['model'].keys())
