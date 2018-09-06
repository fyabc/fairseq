#! /usr/bin/python
# -*- coding: utf-8 -*-

"""The model classifier.

Training:
    Dev data (x, y), eval on M models, get best BLEU on model m: classification task [(x, y) -> m]
Prediction:
    Test data x -> model index m
"""

import torch.nn as nn

from . import utils

__author__ = 'fyabc'


def average_embedding(ckpt_list: list=None):
    if ckpt_list is None:
        ckpt_list = [utils.get_model_path(ckpt) for ckpt in utils.CheckPoints]

    for ckpt_filename in ckpt_list:
        pass


class ModelClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """

        Args:
            x: The batch of source sentence.

        Returns:
            1-D LongTensor. Indicate which model to select for each sentence in the batch.
        """
        pass
