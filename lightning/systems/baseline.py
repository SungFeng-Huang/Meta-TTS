#!/usr/bin/env python3

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l

from .utils import MAML
from utils.tools import get_mask_from_lengths
from lightning.systems.base_adaptor import BaseAdaptorSystem
from lightning.utils import loss2dict


class BaselineSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for baseline FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_learner(self, *args, **kwargs):
        embedding = self.embedding_model.get_new_embedding("table")
        
        emb_layer = nn.Embedding.from_pretrained(embedding, freeze=False, padding_idx=0).to(self.device)
        adapt_dict = nn.ModuleDict({
            k: getattr(self.model, k) for k in self.algorithm_config["adapt"]["modules"]
        })
        adapt_dict["embedding"] = emb_layer
        return MAML(adapt_dict, lr=self.adaptation_lr)
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        return batch
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, "data with 12 elements"

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.

        Function:
            common_step(): Defined in `lightning.systems.system.System`
        """
        loss, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': loss[0], 'losses': loss, 'output': output, '_batch': batch}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, "data with 12 elements"

    def validation_step(self, batch, batch_idx):
        """ Adapted forwarding.

        Function:
            meta_learn(): Defined in `lightning.systems.base_adaptor.BaseAdaptorSystem`
        """
        val_loss, predictions = self.common_step(batch, batch_idx, train=False)
        # val_loss, predictions = self.meta_learn(batch, batch_idx, train=False)
        # qry_batch = batch[0][1][0]
        qry_batch = batch

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}":v for k,v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}

