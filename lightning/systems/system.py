#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from argparse import Namespace

from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor, ModelCheckpoint

from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks import GlobalProgressBar, Saver
from lightning.optimizer import get_optimizer
from lightning.scheduler import get_scheduler
from lightning.utils import LightningMelGAN, loss2dict
from utils.tools import expand, plot_mel


class System(pl.LightningModule):

    default_monitor: str = "val_loss"

    def __init__(self, preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config
        self.save_hyperparameters()

        self.model = FastSpeech2(preprocess_config, model_config, algorithm_config)
        self.loss_func = FastSpeech2Loss(preprocess_config, model_config)

        self.log_dir = log_dir
        self.result_dir = result_dir

        # Although the vocoder is only used in callbacks, we need it to be
        # moved to cuda for faster inference, so it is initialized here with the
        # model, and let pl.Trainer handle the DDP devices.
        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_idx, train=True):
        output = self(*(batch[2:]))
        loss = self.loss_func(batch, output)
        return loss, output

    def training_step(self, batch, batch_idx):
        loss, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': loss[0], 'losses': loss, 'output': output, '_batch': batch}

    def validation_step(self, batch, batch_idx):
        loss, output = self.common_step(batch, batch_idx, train=False)

        loss_dict = {f"Val/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': loss, 'output': output, '_batch': batch}

    def configure_callbacks(self):
        # Checkpoint saver
        save_step = self.train_config["step"]["save_step"]
        checkpoint = ModelCheckpoint(
            monitor="Val/Total Loss", mode="min",
            every_n_train_steps=save_step, save_top_k=-1, save_last=True,
        )

        # Progress bars (step/epoch)
        outer_bar = GlobalProgressBar(process_position=1)
        # inner_bar = ProgressBar(process_position=1) # would * 2

        # Monitor learning rate / gpu stats
        lr_monitor = LearningRateMonitor()
        gpu_monitor = GPUStatsMonitor(
            memory_utilization=True, gpu_utilization=True, intra_step_time=True, inter_step_time=True
        )
        
        # Save figures/audios/csvs
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)

        callbacks = [checkpoint, outer_bar, lr_monitor, gpu_monitor, saver]
        return callbacks

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = get_optimizer(self.model, self.model_config, self.train_config)

        self.scheduler = {
            "scheduler": get_scheduler(self.optimizer, self.train_config),
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            'monitor': self.default_monitor,
        }

        return [self.optimizer], [self.scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if self.local_rank == 0:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                if self.local_rank == 0:
                    print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)


