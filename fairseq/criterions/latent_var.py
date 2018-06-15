# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch

from fairseq import utils

from . import register_criterion
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion('latent_var')
class LatentVarCriterion(LabelSmoothedCrossEntropyCriterion):

    def compute_losses(self, model, net_input, target):
        net_output = model(**net_input)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -lprobs.sum(-1)
        target_pad_mask = (target == self.padding_idx)
        nll_loss[target_pad_mask] = 0
        smooth_loss[target_pad_mask] = 0
        nll_loss = nll_loss.sum(-1)
        smooth_loss = smooth_loss.sum(-1)
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss, smooth_loss


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        k = self.args.latent_category
        bsz = sample['target'].size(0)

        is_training = model.training
        model.eval()
        with torch.no_grad():
            losses = []
            nll_losses = []
            for i in range(k):
                sample['net_input']['latent'] = i
                loss, nll_loss, _ = self.compute_losses(model, sample['net_input'], sample['target'])
                losses.append(loss)
                nll_losses.append(nll_loss)

            losses = torch.cat(losses)
            nll_losses = torch.cat(nll_losses)
            loss, latent = losses.view(k, bsz).min(0)
            nll_loss = nll_losses.view(k, bsz).gather(dim=0, index=latent.unsqueeze(0)).squeeze(0)

            cnt = torch.zeros(k, dtype=torch.long)
            latent_cnt = torch.bincount(latent)
            cnt[:latent_cnt.numel()] = latent_cnt

        if is_training:
            model.train()
            sample['net_input']['latent'] = latent
            loss, nll_loss, _ = self.compute_losses(model, sample['net_input'], sample['target'])

        if reduce:
            loss = loss.sum()
            nll_loss = nll_loss.sum()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
            'latent_count': cnt.data,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        cnt = sum(log.get('latent_count', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
            'latent_count': cnt,
        }
