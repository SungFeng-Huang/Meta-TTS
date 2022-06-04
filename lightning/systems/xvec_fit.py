#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
import torchmetrics
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from collections import Counter
from pytorch_lightning.utilities.cli import instantiate_class


class XvecFitSystem(pl.LightningModule):

    default_monitor: str = "val_loss"

    def __init__(self, model: pl.LightningModule, num_tgt: int,
                 optim_config: dict, scheduler_config: dict):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.num_tgt = num_tgt
        self.model = model
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(num_classes=num_tgt,
                                               average="macro")
        self.train_class_acc = torchmetrics.Accuracy(num_classes=num_tgt,
                                                     average=None)
        self.val_class_acc = torchmetrics.Accuracy(num_classes=num_tgt,
                                                   average=None)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch[4].transpose(1, 2))

        loss = self.loss_func(output, batch[2])
        self.log("train_loss", loss.item(), sync_dist=True)

        self.train_acc(output, batch[2])
        self.log("train_acc", self.train_acc, sync_dist=True, prog_bar=True)

        # forward -> compute -> DDP sync
        self.train_class_acc.update(output, batch[2])
        self.train_count.update(batch[2].cpu().numpy().tolist())

        return {'loss': loss}

    def on_train_epoch_end(self):
        print('\n')
    #     pDropMax = 0.2
    #     stepFrac = 0.5
    #     current_frac = float(self.current_epoch) / self.trainer.max_epochs
    #     if current_frac < stepFrac:
    #         p_drop = pDropMax * current_frac / stepFrac
    #     elif current_frac < (stepFrac + 1) / 2:
    #         scale = ((stepFrac + 1) / 2 - current_frac) / ((1 - stepFrac) / 2)
    #         p_drop = pDropMax * scale   # fast decay to 0
    #     else:
    #         p_drop = 0
    #
    #     self.model.set_dropout_rate(p_drop)

    def on_fit_start(self):
        self.train_count = Counter()

    def on_validation_start(self):
        self.val_count = Counter()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch[4].transpose(1, 2))

        loss = self.loss_func(output, batch[2])
        self.log("val_loss", loss.item(), sync_dist=True)

        self.val_class_acc.update(output, batch[2])
        self.val_count.update(batch[2].cpu().numpy().tolist())

        return {'loss': loss}

    def on_validation_epoch_end(self,):
        self.print(f"Epoch {self.current_epoch}")
        self.print({k: v.item() for k, v in self.trainer.callback_metrics.items()
                    if k in ["train_loss", "train_acc", "val_loss", "val_acc"]})

        data = {
            "train_count": [self.train_count[i] for i in range(self.num_tgt)],
            "val_count": [self.val_count[i] for i in range(self.num_tgt)],
        }
        self.train_count.clear()
        self.val_count.clear()

        try:
            train_acc = self.train_class_acc.compute().cpu().numpy().tolist()
            self.log_dict({f"train_acc/{i}": acc for i, acc in
                           enumerate(train_acc)})
            data.update({"train_acc": train_acc})
        except Exception as e:
            self.print(e)
        self.train_class_acc.reset()

        try:
            val_acc = self.val_class_acc.compute().cpu().numpy().tolist()
            self.log_dict({f"val_acc/{i}": acc for i, acc in
                           enumerate(val_acc)})
            self.log("val_acc", np.mean(val_acc))
            data.update({"val_acc": val_acc})
        except Exception as e:
            self.print(e)
        self.val_class_acc.reset()

        df = pd.DataFrame(data)
        self.print(df.to_string(header=True, index=True))

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = instantiate_class(self.model.parameters(), self.optim_config)

        if self.scheduler_config["class_path"].split('.')[-1] == "OneCycleLR":
            self.scheduler_config["init_args"].update({
                "total_steps": self.trainer.estimated_stepping_batches,
            })
        scheduler = instantiate_class(self.optimizer, self.scheduler_config)
        self.scheduler = {
            "scheduler": scheduler,
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            'monitor': self.default_monitor,
        }

        return [self.optimizer], [self.scheduler]

