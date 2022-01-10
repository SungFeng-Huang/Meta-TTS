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
from lightning.systems.utils import Task
from lightning.utils import loss2dict


class MetaSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # NOTE: `self.model.encoder` and `self.learner.encoder` are pointing to
        # the same variable, they are not two variables with the same values.
        if self.global_step == 0 and self.pretrain_step > 0:
            self.start_pretrain()
        if self.global_step == self.pretrain_step:
            self.end_pretrain()
        
        sup_batch, qry_batch, ref_phn_feats, lang_id = batch[0]
        if self.codebook_type == "table":
            self.model.encoder.src_word_emb._parameters['weight'] = \
                self.model.get_new_embedding(ref_phn_feats, f"table-{lang_id}").clone()
        else:
            self.adaptation_steps = self.algorithm_config["adapt"]["train"]["steps"]
            self.model.encoder.src_word_emb._parameters['weight'] = \
                self.model.get_new_embedding(ref_phn_feats, self.codebook_type).clone()
        return [(sup_batch, qry_batch)]
    
    def start_pretrain(self):
        print("Start pretrain.")
        self.codebook_type = "table"
        self.adaptation_steps = 0

    def end_pretrain(self):
        print("End pretrain.")
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        if codebook_config["type"] == "embedding":
            self.codebook_type = "table"
        elif codebook_config["type"] == "codebook":
            self.codebook_type = self.codebook_config["attention"]["type"]
        else:
            raise NotImplementedError
        self.adaptation_steps = self.algorithm_config["adapt"]["train"]["steps"]

    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, task=None, train=True):
        # TODO: overwrite for supporting SGD and iMAML
        # return super().adapt(batch, adaptation_steps, learner, train)

        # MAML
        # TODO: SGD data reuse for more steps
        if learner is None:
            learner = self.learner.clone()
            learner.train()

        if task is None:
            sup_data = batch[0][0][0]
            qry_data = batch[0][1][0]
            task = Task(sup_data=sup_data,
                        qry_data=qry_data,
                        batch_size=self.algorithm_config["adapt"]["imaml"]["batch_size"]) # The batch size is borrowed from the imaml config.

        first_order = not train
        for step in range(adaptation_steps):
            mini_batch = task.next_batch()

            preds = self.forward_learner(learner, *mini_batch[2:])
            train_error = self.loss_func(mini_batch, preds)
            learner.adapt(
                train_error[0], first_order=first_order,
                allow_unused=False, allow_nograd=True
            )
        return learner

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
