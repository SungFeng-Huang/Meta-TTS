#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from lightning.model import XvecTDNN


class BaseMixin(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        """A base system mixin."""
        super().__init__(*args, **kwargs)

        self.loss_func = torch.nn.CrossEntropyLoss()


class XvecWrapperMixin:
    def __init__(self, *args, **kwargs):
        self.model = XvecTDNN(*args, **kwargs)
        self.num_classes = self.model.num_classes
        super().__init__(*args, **kwargs)


class XvecBaseSystem(BaseMixin, XvecTDNN):
    def __init__(self, *args, **kwargs):
        """A base system for XvecTDNN."""
        super().__init__(*args, **kwargs)


class XvecBaseWrapper(XvecWrapperMixin, BaseMixin):
    def __init__(self, *args, **kwargs):
        """A base system wrapper for XvecTDNN."""
        super().__init__(*args, **kwargs)
