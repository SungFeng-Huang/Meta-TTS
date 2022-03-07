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
            monitor="Train/Total Loss", mode="min", # monitor not used
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
        self.saver = saver

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
        changes = {"skip": [], "drop": [], "replace": [], "miss": []}

        if 'model.speaker_emb.weight' in state_dict:
            # Compatiable with old version
            assert "model.speaker_emb.model.weight" in model_state_dict
            assert "model.speaker_emb.model.weight" not in state_dict
            state_dict["model.speaker_emb.model.weight"] = state_dict.pop("model.speaker_emb.weight")
            changes["replace"].append(["model.speaker_emb.weight", "model.speaker_emb.model.weight"])
            is_changed = True

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if k == "model.speaker_emb.model.weight":
                        # train-clean-100: 247 (spk_id 0 ~ 246)
                        # train-clean-360: 904 (spk_id 247 ~ 1150)
                        # train-other-500: 1160 (spk_id 1151 ~ 2310)
                        # dev-clean: 40 (spk_id 2311 ~ 2350)
                        # test-clean: 39 (spk_id 2351 ~ 2389)
                        if self.preprocess_config["dataset"] == "LibriTTS":
                            # LibriTTS: testing emb already in ckpt
                            # Shape mismatch: version problem
                            # (train-clean-100 vs train-all)
                            assert self.algorithm_config["adapt"]["speaker_emb"] == "table"
                            assert (state_dict[k].shape[0] == 326
                                    and model_state_dict[k].shape[0] == 2390), \
                                f"state_dict: {state_dict[k].shape}, model: {model_state_dict[k].shape}"
                            model_state_dict[k][:247] = state_dict[k][:247]
                            model_state_dict[k][-79:] = state_dict[k][-79:]
                        else:
                            # Corpus mismatch
                            assert state_dict[k].shape[0] in [326, 2390]
                            assert self.algorithm_config["adapt"]["speaker_emb"] == "table"
                            if self.algorithm_config["adapt"]["test"].get(
                                    "avg_train_spk_emb", False
                            ):
                                model_state_dict[k][:] = state_dict[k][:247].mean(dim=0)
                                if self.local_rank == 0:
                                    print("Average training speaker emb as testing emb")
                    # Other testing corpus with different # of spks: re-initialize
                    changes["skip"].append([
                        k, model_state_dict[k].shape, state_dict[k].shape
                    ])
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                changes["drop"].append(k)
                is_changed = True

        for k in model_state_dict:
            if k not in state_dict:
                changes["miss"].append(k)
                is_changed = True

        if self.local_rank == 0:
            print()
            for replaced in changes["replace"]:
                before, after = replaced
                print(f"Replace: {before}\n\t-> {after}")
            for skipped in changes["skip"]:
                k, required_shape, loaded_shape = skipped
                print(f"Skip parameter: {k}, \n" \
                           + f"\trequired shape: {required_shape}, " \
                           + f"loaded shape: {loaded_shape}")
            for dropped in changes["drop"]:
                print(f"Dropping parameter: {dropped}")
                del state_dict[dropped]
            for missed in changes["miss"]:
                print(f"Missing parameter: {missed}")
            print()

        if is_changed:
            checkpoint.pop("optimizer_states", None)


    def on_test_start(self):
        if self.algorithm_config["adapt"]["speaker_emb"] == "table":
            if self.local_rank == 0:
                print("Testing speaker emb")
                print("Before:")
                print(self.model.speaker_emb.model.weight[-39:])

            if (self.preprocess_config["dataset"] == "LibriTTS"
                    and self.algorithm_config["adapt"]["test"].get("avg_train_spk_emb", False)):

                with torch.no_grad():
                    self.model.speaker_emb.model.weight[-39:] = \
                        self.model.speaker_emb.model.weight[:247].mean(dim=0)

            if self.local_rank == 0:
                print("After:")
                print(self.model.speaker_emb.model.weight[-39:])
                print()
