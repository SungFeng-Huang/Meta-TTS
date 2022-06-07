#!/usr/bin/env python3

import math
import torch
import pandas as pd
import numpy as np
from collections import Counter
from typing import Union, Mapping, Optional
from torchmetrics import Accuracy
from pytorch_lightning.utilities.cli import instantiate_class
from lightning.model import XvecTDNN
from .validate import XvecValidateSystem


class XvecFitSystem(XvecValidateSystem):

    default_monitor: str = "val_loss"

    def __init__(self,
                 optim_config: dict,
                 lr_sched_config: dict,
                 *args,
                 num_classes: Optional[int] = None,
                 model: Optional[Union[Mapping, XvecTDNN]] = None,
                 model_dict: Optional[Mapping] = None,
                 **kwargs):
        """A system wrapper for fitting XvecTDNN.

        Args:
            model (optional): X-vector model. If specified, `model_dict` would
                be ignored.
            model_dict (optional): X-vector model config. If 'model' is not
                specified, would use `model_dict` instead.
            num_classes: Number of speakers/regions/accents.
            optim_config: Config of optimizer.
            lr_sched_config: Config of learning rate scheduler.
        """
        super().__init__(num_classes=num_classes,
                         model=model,
                         model_dict=model_dict)

        self.optim_config = optim_config
        self.lr_sched_config = lr_sched_config

        self.train_acc = Accuracy(num_classes=self.num_classes,
                                  average="macro")
        self.train_class_acc = Accuracy(num_classes=self.num_classes,
                                        average=None)

    def on_train_start(self):
        self.train_count = Counter()

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

        # Hard-code dropout scheduler
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

    def _get_train_epoch_end_data(self):
        data = {}
        try:
            data["train_count"] = [
                self.train_count[i] for i in range(self.num_classes)]
            self.train_count.clear()
        except:
            pass

        try:
            train_acc = self.train_class_acc.compute().cpu().numpy().tolist()
            data.update({"train_acc": train_acc})
        except Exception as e:
            self.print(e)
        self.train_class_acc.reset()

        return data

    def on_validation_epoch_end(self,):
        self.print(f"Epoch {self.current_epoch}")

        data = self._get_train_epoch_end_data()
        data.update(self._get_val_epoch_end_data())

        if "train_acc" in data:
            train_acc = data["train_acc"]
            self.log_dict({f"train_acc/{i}": acc
                           for i, acc in enumerate(train_acc)})
        if "val_acc" in data:
            val_acc = data["val_acc"]
            self.log_dict({f"val_acc/{i}": acc
                           for i, acc in enumerate(val_acc)})
            self.log("val_acc",
                     np.mean([acc for acc in val_acc
                              if not math.isnan(acc)]))

        df = pd.DataFrame(data)
        self.print({k: v.item() for k, v in self.trainer.callback_metrics.items()
                    if k in ["train_loss", "train_acc", "val_loss", "val_acc"]})
        self.print(df.to_string(header=True, index=True))

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = instantiate_class(self.model.parameters(), self.optim_config)

        if issubclass(eval(self.lr_sched_config["class_path"]),
                      torch.optim.lr_scheduler.OneCycleLR):
            self.lr_sched_config["init_args"].update({
                "total_steps": self.trainer.estimated_stepping_batches,
            })
        scheduler = instantiate_class(self.optimizer, self.lr_sched_config)
        self.scheduler = {
            "scheduler": scheduler,
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            'monitor': self.default_monitor,
        }

        return [self.optimizer], [self.scheduler]

