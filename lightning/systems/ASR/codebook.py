#!/usr/bin/env python3

from lightning.model.asr_model import ASRRefHead

from ..system2 import System
from lightning.model.loss import PhonemeClassificationLoss
from lightning.callbacks.asr_saver import Saver
from lightning.utils import asr_loss2dict as loss2dict


class CodebookSystem(System):
    """
    Concrete class of ASR head for codebook quality evaluation (w/ average).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # tests
        self.test_list = {
        }

    def build_model(self):
        self.model = ASRRefHead(self.model_config, self.algorithm_config)
        self.loss_func = PhonemeClassificationLoss()

    def build_optimized_model(self):
        return self.model

    def build_saver(self):
        return Saver(self.preprocess_config, self.log_dir, self.result_dir)
    
    def common_step(self, batch, batch_idx, train=True):
        _, _, ref, lang_id = batch[0]
        qry_batch = batch[0][1][0]
        predictions = self.model(qry_batch, ref, lang_id)

        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions
    
    def training_step(self, batch, batch_idx):
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss, 'losses': train_loss, 'output': predictions, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.common_step(batch, batch_idx)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}
