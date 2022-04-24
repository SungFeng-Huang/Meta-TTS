#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l

from utils.tools import get_mask_from_lengths
from lightning.systems.new_base_adaptor import BaseAdaptorSystem
from lightning.utils import loss2dict


class BaselineSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        assert len(batch) == 12, "data with 12 elements"

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        """ Normal forwarding.

        Function:
            common_step(): Defined in `lightning.systems.system.System`
        """
        loss, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=len(batch[0]))
        return {'loss': loss[0],
                'losses': [l.detach() for l in loss],
                'output': [o.detach() for o in output],
                '_batch': batch}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self._on_meta_batch_start(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ Adapted forwarding.

        Function:
            meta_learn(): Defined in `lightning.systems.base_adaptor.BaseAdaptorSystem`
        """
        val_loss, predictions = self.meta_learn(batch, batch_idx, train=False)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}":v for k,v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=1)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}

