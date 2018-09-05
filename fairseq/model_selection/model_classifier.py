#! /usr/bin/python
# -*- coding: utf-8 -*-

"""The model classifier.

Training:
    Dev data (x, y), eval on M models, get best BLEU on model m: classification task [(x, y) -> m]
Prediction:
    Test data x -> model index m
"""

import torch.nn as nn

__author__ = 'fyabc'


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
