#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate model selection classifier data.

[NOTE]: Valid and test set are combined in the result file.
Order: [valid, test].
"""

import numpy as np

from fairseq.data import dictionary
from fairseq.data import IndexedDataset
from fairseq.model_selection.utils import CheckPoints, Subsets, TargetFiles, get_data_path, get_dict_path, get_bleu_path

__author__ = 'fyabc'


def read_dataset(subset, lang):
    dict_path = get_dict_path(lang)
    data_path = get_data_path(subset, lang)

    dict_ = dictionary.Dictionary.load(dict_path)
    data = IndexedDataset(data_path, fix_lua_indexing=True)

    return [dict_.string(tensor_line) for tensor_line in data]


def main():
    lang = 'en'

    with open(TargetFiles['x'], 'w', encoding='utf-8') as f_x, open(TargetFiles['y'], 'w', encoding='utf-8') as f_y:
        for subset in Subsets:
            print('Generating {} data...'.format(subset), end='')
            dataset = read_dataset(subset, lang)
            scores = np.zeros([len(dataset), len(CheckPoints)], dtype=np.float64)
            for i, ckpt in enumerate(CheckPoints):
                bleu_path = get_bleu_path(subset, ckpt)
                bleu_list = []
                with open(bleu_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            bleu_list.append(float(line))
                scores[:, i] = bleu_list

            labels = []
            for sen_scores in scores:
                candidates = np.argwhere(sen_scores == np.max(sen_scores)).flatten()
                labels.append(np.random.choice(candidates))

            for sentence, label in zip(dataset, labels):
                print(sentence, file=f_x)
                print(label, file=f_y)

            print(' done')

    print('Dump inputs to {}, targets to {}.'.format(TargetFiles['x'], TargetFiles['y']))


if __name__ == '__main__':
    main()
