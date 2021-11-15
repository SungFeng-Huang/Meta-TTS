#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l

from learn2learn.algorithms.lightning import LightningMAML

from utils.tools import get_mask_from_lengths
from lightning.systems.base_adaptor import BaseAdaptorSystem
from lightning.utils import loss2dict


class MetaSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def _on_meta_batch_start(self, batch):
        # """ Check meta-batch data """
        # # NOTE: should rename `ref_p_embedding` to something like
        # # `ref_phn_feats` or `ref_phn_representations`
        # assert len(batch) == 1, "meta_batch_per_gpu"
        # assert len(batch[0]) == 3, "sup + qry + ref_p_embedding"
        # assert len(batch[0][0]) == 1, "n_batch == 1"
        # assert len(batch[0][0][0]) == 12, "data with 12 elements"

    # def on_after_batch_transfer(self, batch, dataloader_idx):
        # # NOTE: `self.model.encoder` and `self.learner.encoder` are pointing to
        # # the same variable, they are not two variables with the same values.
        # ref_phn_feats = batch[0][2]
        # # self.model.encoder.src_word_emb.set_quantize_matrix(ref_phn_feats)
        # # TODO
        # self.model.encoder.src_word_emb._parameter['weight'] = emb_tensor.clone()
        # del batch[0][2]
        # return batch

    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, train=True):
        # TODO: overwrite for supporting SGD and iMAML
        return super().adapt(batch, adaptation_steps, learner, train)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
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
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0], 'losses': train_loss, 'output': predictions, '_batch': qry_batch}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)

    def validation_step(self, batch, batch_idx):
        """ Adapted forwarding.

        Function:
            meta_learn(): Defined in `lightning.systems.base_adaptor.BaseAdaptorSystem`
        """
        val_loss, predictions = self.meta_learn(batch, batch_idx)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}

