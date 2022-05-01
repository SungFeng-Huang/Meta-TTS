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

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        assert len(batch) == 12, "data with 12 elements"

    def training_step(self, batch, batch_idx):
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
