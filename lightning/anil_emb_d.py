#!/usr/bin/env python3

import os
import json
import pytorch_lightning as pl
import torch
import numpy as np
import learn2learn as l2l
from tqdm import tqdm
from pytorch_lightning.loggers.base import merge_dicts
from torch.utils.data import DataLoader

from learn2learn.algorithms.lightning import LightningMAML
from learn2learn.utils.lightning import EpisodicBatcher

from model import FastSpeech2Loss, FastSpeech2
from utils.tools import get_mask_from_lengths
from lightning.base_emb_d import BaselineEmbDecSystem
from lightning.collate import get_meta_collate
from lightning.utils import seed_all


class ANILEmbDecSystem(BaselineEmbDecSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_ways     = self.train_config["meta"]["ways"]
        self.train_shots    = self.train_config["meta"]["shots"]
        self.train_queries  = self.train_config["meta"]["queries"]
        assert self.train_ways == self.test_ways, \
            "For ANIL, train_ways should be equal to test_ways."

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def training_step(self, batch, batch_idx):

        train_loss, predictions = self.meta_learn(
            batch, batch_idx, self.train_ways, self.train_shots, self.train_queries
        )

        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]

        # Synthesis one sample
        if (self.global_step+1) % self.train_config["step"]["synth_step"] == 0 and self.local_rank == 0:
            qry_batch = batch[0][1][0]
            metadata = {'stage': "Training", 'sup_ids': sup_ids}
            fig, wav_reconstruction, wav_prediction, basename = self.synth_one_sample(
                qry_batch, predictions
            )
            step = self.global_step+1
            self.log_figure(
                f"Training/step_{step}_{basename}", fig
            )
            self.log_audio(
                f"Training/step_{step}_{basename}_reconstructed",
                wav_reconstruction, metadata=metadata
            )
            self.log_audio(
                f"Training/step_{step}_{basename}_synthesized",
                wav_prediction, metadata=metadata
            )

        # Log message to log.txt and print to stdout
        if (self.global_step+1) % self.trainer.log_every_n_steps == 0:
            message = f"Step {self.global_step+1}/{self.trainer.max_steps}, "
            message += self.loss2str(train_loss)
            self.log_text(message)

        # Log metrics to CometLogger
        comet_log_dict = {f"train_{k}":v for k,v in self.loss2dict(train_loss).items()}
        self.log_dict(comet_log_dict, sync_dist=True)
        return train_loss[0]

    def train_dataloader(self):
        # Epochify task dataset, for periodic validation
        meta_batch_size = self.train_config["meta"]["meta_batch_size"]
        val_step = self.train_config["step"]["val_step"]
        epoch_length = meta_batch_size*val_step

        train_task_dataset = self._few_shot_task_dataset(
            self.train_dataset, self.train_ways, self.train_shots, self.train_queries,
            task_per_speaker=-1, epoch_length=epoch_length
        ).train_dataloader()

        # DataLoader, batch_size = batch size on each gpu
        self.train_loader = DataLoader(
            train_task_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda batch: batch,
            num_workers=8,
        )
        return self.train_loader

    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)

    def validation_epoch_end(self, val_outputs=None):
        super().validation_epoch_end(val_outputs)

    def val_dataloader(self):
        return super().val_dataloader()

    def on_save_checkpoint(self, checkpoint):
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)


class ANILEmb1DecSystem(ANILEmbDecSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(
        self,
        model=None,
        optimizer=None,
        loss_func=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        scheduler=None,
        configs=None,
        vocoder=None,
        log_dir=None,
        result_dir=None,
     ):
        preprocess_config, model_config, train_config = configs

        if model is None:
            model = FastSpeech2(preprocess_config, model_config)
        if loss_func is None:
            loss_func = FastSpeech2Loss(preprocess_config, model_config)

        if model_config["multi_speaker"]:
            del model.speaker_emb
            model.speaker_emb = torch.nn.Embedding(
                1,
                model_config["transformer"]["encoder_hidden"],
            )
        
        del optimizer
        del scheduler
        optimizer = None
        scheduler = None
        super().__init__(
            model, optimizer, loss_func,
            train_dataset, val_dataset, test_dataset,
            scheduler, configs, vocoder,
            log_dir, result_dir
        )

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        with torch.no_grad():
            batch[0][0][0][2].zero_()
            batch[0][1][0][2].zero_()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        with torch.no_grad():
            batch[0][0][0][2].zero_()
            batch[0][1][0][2].zero_()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        with torch.no_grad():
            batch[0][0][0][2].zero_()
            batch[0][1][0][2].zero_()
