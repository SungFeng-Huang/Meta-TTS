#!/usr/bin/env python3

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l

from learn2learn.algorithms.lightning import LightningMAML

from text.define import LANG_ID2SYMBOLS
from .utils import MAML
from utils.tools import get_mask_from_lengths
from lightning.systems.base_adaptor import BaseAdaptorSystem
from lightning.systems.utils import Task
from lightning.utils import loss2dict


class DualSystem(BaseAdaptorSystem):
    """
    No adapt + asr cycle consistency loss for PhonemeEmbedding banks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_codebook_type()
        self.cycle_loss_func = None
        self.asr_projs = nn.ModuleDict({
            f"proj-{lang_id}": nn.Linear(self.model_config["transformer"]["encoder_hidden"], len(v)) 
            for lang_id, v in LANG_ID2SYMBOLS.items() if len(v) > 0
        })
    
    def init_codebook_type(self):        
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        if codebook_config["type"] == "embedding":
            self.codebook_type = "table-sep"
        elif codebook_config["type"] == "codebook":
            self.codebook_type = codebook_config["attention"]["type"]
        else:
            raise NotImplementedError
        self.adaptation_steps = self.algorithm_config["adapt"]["train"]["steps"]

    def build_learner(self, batch):
        _, _, ref_phn_feats, lang_id = batch[0]
        embedding, asr_embedding = self.embedding_model.get_new_embedding(
            self.codebook_type, ref_phn_feats=ref_phn_feats, 
            lang_id=lang_id, return_asr=True
        )
        emb_layer = nn.Embedding.from_pretrained(embedding, freeze=False, padding_idx=0).to(self.device)
        self.asr_emb_layer = nn.Embedding.from_pretrained(asr_embedding, freeze=False, padding_idx=0).to(self.device)
        self.asr_proj = self.asr_projs[f"proj-{lang_id}"]
        adapt_dict = nn.ModuleDict({
            k: getattr(self.model, k) for k in self.algorithm_config["adapt"]["modules"]
        })
        adapt_dict["embedding"] = emb_layer
        return MAML(adapt_dict, lr=self.adaptation_lr)
    
    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, task=None, train=True):
        # TODO: overwrite for supporting SGD and iMAML
        # return super().adapt(batch, adaptation_steps, learner, train)

        # MAML
        # TODO: SGD data reuse for more steps
        if learner is None:
            learner = self.build_learner(batch)
            sup_batch, qry_batch, _, _ = batch[0]
            batch = [(sup_batch, qry_batch)]
            # learner = self.learner.clone()
            learner = learner.clone()
            learner.train()

        if task is None:
            sup_data = batch[0][0][0]
            qry_data = batch[0][1][0]
            task = Task(sup_data=sup_data,
                        qry_data=qry_data,
                        batch_size=self.algorithm_config["adapt"]["imaml"]["batch_size"]) # The batch size is borrowed from the imaml config.

        # no adapt when training
        first_order = not train
        for step in range(adaptation_steps):
            mini_batch = task.next_batch()

            preds = self.forward_learner(learner, *mini_batch[2:])
            train_error = self.loss_func(mini_batch, preds)
            learner.adapt_(
                train_error[0], first_order=first_order,
                allow_unused=False, allow_nograd=True
            )
        return learner

    def meta_learn(self, batch, batch_idx, train=True):
        learner = self.adapt(batch, min(self.adaptation_steps, self.test_adaptation_steps), train=train)

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]

        # Evaluating the adapted model
        # Use speaker embedding of support set
        # predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:], average_spk_emb=True)
        predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
        valid_error = self.loss_func(qry_batch, predictions)
        if train:
            asr_predictions = self.asr_classifier(qry_batch)
            cycle_loss = self.cycle_loss_func(qry_batch, asr_predictions)
            valid_error = (valid_error[0] + cycle_loss, *valid_error[1:], cycle_loss)
        return valid_error, predictions

    def asr_classifier(self, qry_batch):
        # Input phonemes id, output cycle consistency loss by using asr_embedding.
        # This indicates how well does asr_embedding separate unseen phonemes.

        emb_texts = self.asr_emb_layer(qry_batch[3])
        return self.asr_proj(emb_texts)

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
