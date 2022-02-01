#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from lightning.model.asr_model import ASRRefHead

from utils.tools import get_mask_from_lengths
from ..system2 import System
from lightning.systems.utils import Task
from lightning.utils import loss2dict, LightningMelGAN
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks import Saver


class CodebookSystem(System):
    """
    Concrete class of ASR head for codebook quality evaluation (w/ average).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # tests
        self.test_list = {
        }
        #TODO
        # loss func and saver

    def build_model(self):
        self.model = ASRRefHead(self.model_config, self.algorithm_config)
        self.loss_func = ()

    def build_optimized_model(self):
        return self.model

    def build_saver(self):
        pass
        # saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        # saver.set_baseline_saver()
        # return saver
    
    def common_step(self, batch, batch_idx, train=True):
        sup_batch, qry_batch, ref, lang_id = batch[0]
        predictions = self.model(qry_batch, ref, lang_id)

        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions
    
    def training_step(self, batch, batch_idx):
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0], 'losses': train_loss, 'output': predictions, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.common_step(batch, batch_idx)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}
