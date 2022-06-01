#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")


class XvecFitSystem(pl.LightningModule):

    default_monitor: str = "val_loss"

    def __init__(self, model, total_steps):
        super().__init__()
        self.save_hyperparameters()

        self.total_steps = total_steps

        self.model = model
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_idx, train=True):
        output = self(*(batch[4]))
        loss = self.loss_func(batch[2], output)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        self.log("train_loss", loss, sync_dist=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        pDropMax = 0.2
        stepFrac = 0.5
        current_frac = float(self.current_epoch) / self.trainer.max_epochs
        if current_frac < stepFrac:
            p_drop = pDropMax * current_frac / stepFrac
        elif current_frac < (stepFrac + 1) / 2:
            scale = ((stepFrac + 1) / 2 - current_frac) / ((1 - stepFrac) / 2)
            p_drop = pDropMax * scale   # fast decay to 0
        else:
            p_drop = 0

        self.model.set_dropout_rate(p_drop)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.common_step(batch, batch_idx, train=False)

        self.log("val_loss", loss, sync_dist=True)
        return {'loss': loss}

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=2e-3,
            cycle_momentum=False,
            div_factor=5,
            final_div_factor=1e+3,
            total_steps=self.total_steps,
            pct_start=0.15
        )
        self.scheduler = {
            "scheduler": scheduler,
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            'monitor': self.default_monitor,
        }

        return [self.optimizer], [self.scheduler]

