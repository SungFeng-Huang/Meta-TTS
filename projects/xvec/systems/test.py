#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import Counter
from torchmetrics import Accuracy

from ..model import XvecTDNN
from .base import BaseMixin, XvecWrapperMixin


class TestMixin(BaseMixin):

    def __init__(self,
                 *args,
                 **kwargs):
        """A system mixin for testing.

        Args:
            (*args, **kwargs): Arguments for XvecTDNN.
        """
        super().__init__(*args, **kwargs)

        self.test_class_acc = Accuracy(num_classes=self.num_classes,
                                       average=None, ignore_index=11)

    def on_test_start(self):
        self.test_count = Counter()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch["mels"].transpose(1, 2))

        loss = self.loss_func(output, batch["target"])
        self.log("test_loss", loss.item(), sync_dist=True)

        self.test_class_acc.update(output, batch["target"])
        self.test_count.update(batch["target"].cpu().numpy().tolist())

        return {'loss': loss}

    def on_test_epoch_end(self,):
        self.print(f"Epoch {self.current_epoch}")

        data = {
            "test_count": [self.test_count[i] for i in range(self.num_classes)],
        }
        self.test_count.clear()

        try:
            test_acc = self.test_class_acc.compute().cpu().numpy().tolist()
            self.log_dict({f"test_acc/{i}": acc
                           for i, acc in enumerate(test_acc)
                           if not np.isnan(acc)})
            self.log("test_acc",
                     np.mean([acc for acc in test_acc
                              if not np.isnan(acc)]))
            data.update({"test_acc": test_acc})
        except Exception as e:
            self.print(e)
        self.test_class_acc.reset()

        df = pd.DataFrame(data)
        self.print({k: v.item() for k, v in self.trainer.callback_metrics.items()
                    if k in ["test_loss", "test_acc"]})
        self.print(df.to_string(header=True, index=True))


class XvecTestSystem(TestMixin, XvecTDNN):
    def __init__(self, *args, **kwargs):
        """A test system for XvecTDNN.
        """
        super().__init__(*args, **kwargs)


class XvecTestWrapper(XvecWrapperMixin, TestMixin):
    def __init__(self, *args, **kwargs):
        """A test wrapper for XvecTDNN.
        """
        super().__init__(*args, **kwargs)
