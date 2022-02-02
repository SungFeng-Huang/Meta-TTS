#!/usr/bin/env python3

import torch
from lightning.model.asr_model import ASRHead

from ..system2 import System
from lightning.model.loss import PhonemeClassificationLoss
from lightning.callbacks.asr_saver import Saver
from lightning.utils import asr_loss2dict as loss2dict


class BaselineSystem(System):
    """
    Concrete class of ASR head for codebook quality evaluation (w/o average).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # tests
        self.test_list = {
        }

    def build_model(self):
        self.model = ASRHead(self.model_config, self.algorithm_config)
        self.loss_func = PhonemeClassificationLoss()

    def build_optimized_model(self):
        return self.model

    def build_saver(self):
        return Saver(self.preprocess_config, self.log_dir, self.result_dir)
    
    def common_step(self, batch, batch_idx, train=True):
        _batch, ref, lang_id = batch[0]
        predictions = self.model(_batch, ref, lang_id)

        loss = self.loss_func(_batch, predictions)
        return loss, predictions
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 1, "data with 12 elements"
        assert len(batch[0]) == 3, "data with 12 elements"
        assert len(batch[0][0]) == 12, "data with 12 elements"
    
    def training_step(self, batch, batch_idx):
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss, 'losses': train_loss, 'output': predictions, '_batch': batch[0][0], 'lang_id': batch[0][2]}

    def on_val_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, "data with 12 elements"
    
    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.common_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': batch[0][0], 'lang_id': batch[0][2]}

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        outputs = {}
        for test_name, test_fn in getattr(self, "test_list", {}).items(): 
            outputs[test_name] = test_fn(batch, batch_idx)

        return outputs
