#!/usr/bin/env python3

import os
import torch
import pandas as pd
from pytorch_lightning.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelCheckpoint
# from learn2learn.algorithms import MAML

# from lightning.systems.system import System
from lightning.systems.base_adapt.base import BaseAdaptorSystem
from lightning.callbacks.saver import FitSaver
from lightning.optimizer import get_optimizer
from lightning.scheduler import get_scheduler
from lightning.utils import loss2dict


VERBOSE = False

class BaseAdaptorFitSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, preprocess_config, model_config, train_config,
                 algorithm_config, log_dir=None, result_dir=None):
        super().__init__(preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir)

        self.adaptation_steps = self.algorithm_config["adapt"]["train"]["steps"]

    def configure_callbacks(self):
        # Checkpoint saver
        save_step = self.train_config["step"]["save_step"]
        checkpoint = ModelCheckpoint(
            monitor="Train/Total Loss", mode="min", # monitor not used
            every_n_train_steps=save_step, save_top_k=-1, save_last=True,
        )

        # Monitor learning rate / gpu stats
        lr_monitor = LearningRateMonitor()
        gpu_monitor = DeviceStatsMonitor()

        # Save figures/audios/csvs
        saver = FitSaver(self.preprocess_config, self.log_dir, self.result_dir)
        self.saver = saver

        callbacks = [checkpoint, lr_monitor, gpu_monitor, saver]
        return callbacks

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = get_optimizer(self.model, self.model_config, self.train_config)

        self.scheduler = {
            "scheduler": get_scheduler(self.optimizer, self.train_config),
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            # 'monitor': self.default_monitor,
        }

        return [self.optimizer], [self.scheduler]

    def on_train_batch_start(self, batch, batch_idx):
        """Log global_step to progress bar instead of GlobalProgressBar
        callback.
        """
        self.log("global_step", self.global_step, prog_bar=True)

    def on_train_epoch_end(self):
        """Print logged metrics during on_train_epoch_end()."""
        key_map = {
            "Train/Total Loss": "total_loss",
            "Train/Mel Loss": "mel_loss",
            "Train/Mel-Postnet Loss": "_mel_loss",
            "Train/Duration Loss": "d_loss",
            "Train/Pitch Loss": "p_loss",
            "Train/Energy Loss": "e_loss",
        }
        loss_dict = {"step": self.global_step}
        for k in key_map:
            loss_dict[key_map[k]] = self.trainer.callback_metrics[k].item()

        df = pd.DataFrame([loss_dict]).set_index("step")
        if (self.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0:
            self.print(df.to_string(header=True, index=True))
        else:
            self.print(df.to_string(header=True, index=True).split('\n')[-1])

    def on_validation_start(self,):
        self.mem_stats = torch.cuda.memory_stats(self.device)
        self.csv_path = os.path.join(self.log_dir, "csv",
                                     f"validate-{self.global_step}.csv")
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ LightningModule interface.
        Operates on a single batch of data from the validation set.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
        """ validation_step() during Trainer.fit() """
        _, val_loss, predictions = self.meta_learn(batch, self.adaptation_steps, train=False)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=1)
        return {
            'losses': val_loss,
            'output': predictions,
            # '_batch': batch[0][1][0],
        }

    def on_validation_end(self):
        """ LightningModule interface.
        Called at the end of validation.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
        """ on_validation_end() during Trainer.fit() """
        if self.global_step > 0:
            self.print(f"Step {self.global_step} - Val losses")
            loss_dicts = []
            for i in range(len(self.trainer.datamodule.val_datasets)):
                key_map = {
                    f"Val/Total Loss/dataloader_idx_{i}": "total_loss",
                    f"Val/Mel Loss/dataloader_idx_{i}": "mel_loss",
                    f"Val/Mel-Postnet Loss/dataloader_idx_{i}": "_mel_loss",
                    f"Val/Duration Loss/dataloader_idx_{i}": "d_loss",
                    f"Val/Pitch Loss/dataloader_idx_{i}": "p_loss",
                    f"Val/Energy Loss/dataloader_idx_{i}": "e_loss",
                }
                loss_dict = {"dataloader_idx": int(i)}
                for k in key_map:
                    loss_dict[key_map[k]] = self.trainer.callback_metrics[k].item()
                loss_dicts.append(loss_dict)

            df = pd.DataFrame(loss_dicts).set_index("dataloader_idx")
            self.print(df.to_string(header=True, index=True)+'\n')
