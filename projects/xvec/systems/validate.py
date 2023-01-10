#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import Counter
from torchmetrics import Accuracy

from ..model import XvecTDNN
from .base import BaseMixin, XvecWrapperMixin


class ValidateMixin(BaseMixin):

    def __init__(self, *args, **kwargs):
        """A system mixin for validating.

        Args:
            (*args, **kwargs): Arguments for XvecTDNN.
        """
        super().__init__(*args, **kwargs)

        self.val_classwise_acc = Accuracy(num_classes=self.num_classes,
                                      average=None, ignore_index=-100)
                                      # average=None, ignore_index=11)

    def on_validation_start(self):
        self.val_count = Counter()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch["mels"].transpose(1, 2))

        loss = self.loss_func(output, batch["target"])
        self.log("val_loss", loss.item(), sync_dist=True)

        self.val_classwise_acc.update(output, batch["target"])
        self.val_count.update(batch["target"].cpu().numpy().tolist())

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
            val_acc = self.val_classwise_acc.compute().cpu().numpy().tolist()
            data.update({"val_acc": val_acc})
        except Exception as e:
            self.print(e)
        self.val_classwise_acc.reset()

        return data

    def on_validation_epoch_end(self,):
        self.print(f"Epoch {self.current_epoch}")

        data = self._get_val_epoch_end_data()
        if "val_acc" in data:
            val_acc = data["val_acc"]
            val_count = data["val_count"]
            self.log_dict({f"val_acc/{i}": acc
                           for i, acc in enumerate(val_acc)
                           if not np.isnan(acc)})
            self.log("val_acc",
                     np.mean([
                         acc for i, acc in enumerate(val_acc)
                         if not np.isnan(acc) and val_count[i] > 0
                     ]))

        df = pd.DataFrame(data)
        self.print({k: v.item() for k, v in self.trainer.callback_metrics.items()
                    if k in ["train_loss", "train_acc", "val_loss", "val_acc"]})
        if self.num_classes < 20:
            self.print(df.to_string(header=True, index=True))
        else:
            self.print("Top 10 val_acc:")
            self.print(df.nlargest(n=10, columns=["val_acc"], keep='all')
                       .to_string(header=True, index=True))
            self.print("Last 10 val_acc:")
            self.print(df.nsmallest(n=10, columns=["val_acc"], keep='all')
                       .to_string(header=True, index=True))
            self.print("Statistics of val_acc:")
            self.print(df["val_acc"].value_counts(sort=False, bins=20)
                       .to_string(header=True, index=True))


class XvecValidateSystem(ValidateMixin, XvecTDNN):
    def __init__(self, *args, **kwargs):
        """A validation system for XvecTDNN.
        """
        super().__init__(*args, **kwargs)


class XvecValidateWrapper(XvecWrapperMixin, ValidateMixin):
    def __init__(self, *args, **kwargs):
        """A validation wrapper for XvecTDNN.
        """
        super().__init__(*args, **kwargs)
