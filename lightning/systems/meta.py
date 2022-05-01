#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l

from learn2learn.algorithms.lightning import LightningMAML

from utils.tools import get_mask_from_lengths
from lightning.systems.new_base_adaptor import BaseAdaptorSystem
from lightning.utils import loss2dict


class MetaSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        if self.algorithm_config["adapt"]["phoneme_emb"]["type"] == "codebook":
            # NOTE: `self.model.encoder` and `self.learner.encoder` are pointing to
            # the same variable, they are not two variables with the same values.
            sup_batch, qry_batch, ref_phn_feats = batch[0]
            self.model.encoder.src_word_emb._parameters['weight'] = \
                self.model.phn_emb_generator.get_new_embedding(ref_phn_feats).clone()
            return [(sup_batch, qry_batch)]
        else:
            return batch

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self._on_meta_batch_start(batch)

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.

        Function:
            common_step(): Defined in `lightning.systems.system.System`
        """
        train_loss, predictions = self.meta_learn(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=1)
        return {'loss': train_loss[0],
                'losses': [loss.detach() for loss in train_loss],
                'output': [p.detach() for p in predictions],
                '_batch': qry_batch}
