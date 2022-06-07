#!/usr/bin/env python3

import math
import pandas as pd
import numpy as np
from collections import Counter
from typing import Union, Mapping, Optional
from torchmetrics import Accuracy
from lightning.model import XvecTDNN
from .base import XvecBaseSystem


class XvecValidateSystem(XvecBaseSystem):

    def __init__(self,
                 *args,
                 num_classes: Optional[int] = None,
                 model: Optional[Union[Mapping, XvecTDNN]] = None,
                 model_dict: Optional[Mapping] = None,
                 **kwargs):
        """A system wrapper for validating XvecTDNN.

        Args:
            model (optional): X-vector model. If specified, `model_dict` would
                be ignored.
            model_dict (optional): X-vector model config. If 'model' is not
                specified, would use `model_dict` instead.
            num_classes: Number of speakers/regions/accents.
        """
        super().__init__(num_classes=num_classes,
                         model=model,
                         model_dict=model_dict)

        self.val_class_acc = Accuracy(num_classes=self.num_classes,
                                      average=None)

    def on_validation_start(self):
        self.val_count = Counter()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch[4].transpose(1, 2))

        loss = self.loss_func(output, batch[2])
        self.log("val_loss", loss.item(), sync_dist=True)

        self.val_class_acc.update(output, batch[2])
        self.val_count.update(batch[2].cpu().numpy().tolist())

        return {'loss': loss}

    def _get_val_epoch_end_data(self):
        data = {}

        try:
            data["val_count"] = [
                self.val_count[i] for i in range(self.num_classes)]
            self.val_count.clear()
        except:
            pass

        try:
            val_acc = self.val_class_acc.compute().cpu().numpy().tolist()
            data.update({"val_acc": val_acc})
        except Exception as e:
            self.print(e)
        self.val_class_acc.reset()

        return data

    def on_validation_epoch_end(self,):
        self.print(f"Epoch {self.current_epoch}")

        data = self._get_val_epoch_end_data()
        if "val_acc" in data:
            val_acc = data["val_acc"]
            self.log_dict({f"val_acc/{i}": acc
                           for i, acc in enumerate(val_acc)
                           if not math.isnan(acc)})
            self.log("val_acc",
                     np.mean([acc for acc in val_acc
                              if not math.isnan(acc)]))

        df = pd.DataFrame(data)
        self.print({k: v.item() for k, v in self.trainer.callback_metrics.items()
                    if k in ["train_loss", "train_acc", "val_loss", "val_acc"]})
        self.print(df.to_string(header=True, index=True))
