#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import Counter
from typing import Union, Mapping, Optional
from torchmetrics import Accuracy
from lightning.model import XvecTDNN
from .base import XvecBaseSystem


class XvecTestSystem(XvecBaseSystem):

    def __init__(self,
                 *args,
                 num_classes: Optional[int] = None,
                 model: Optional[Union[Mapping, XvecTDNN]] = None,
                 model_dict: Optional[Mapping] = None,
                 **kwargs):
        """A system wrapper for testing XvecTDNN.

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

        self.test_class_acc = Accuracy(num_classes=self.num_classes,
                                       average=None)

    def on_test_start(self):
        self.test_count = Counter()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch[4].transpose(1, 2))

        loss = self.loss_func(output, batch[2])
        self.log("test_loss", loss.item(), sync_dist=True)

        self.test_class_acc.update(output, batch[2])
        self.test_count.update(batch[2].cpu().numpy().tolist())

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
                           for i, acc in enumerate(test_acc)})
            self.log("test_acc", np.mean(test_acc))
            data.update({"test_acc": test_acc})
        except Exception as e:
            self.print(e)
        self.test_class_acc.reset()

        df = pd.DataFrame(data)
        self.print({k: v.item() for k, v in self.trainer.callback_metrics.items()
                    if k in ["test_loss", "test_acc"]})
        self.print(df.to_string(header=True, index=True))

