#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from typing import Union, Mapping, Optional
from pytorch_lightning.utilities.cli import instantiate_class
from lightning.model import XvecTDNN


class XvecBaseSystem(pl.LightningModule):

    def __init__(self,
                 num_classes: Optional[int] = None,
                 model: Optional[Union[Mapping, XvecTDNN]] = None,
                 model_dict: Optional[Mapping] = None):
        """A system wrapper for fitting XvecTDNN.

        Args:
            model (optional): X-vector model. If specified, `model_dict` would
                be ignored.
            model_dict (optional): X-vector model config. If 'model' is not
                specified, would use `model_dict` instead.
        """
            # num_classes: Number of speakers/regions/accents.
        super().__init__()
        self.save_hyperparameters()

        if model is None:
            if model_dict is None:
                raise TypeError
            model = model_dict

        if isinstance(model, Mapping):
            num_classes = model["init_args"].get("num_classes", num_classes)
            if num_classes is None:
                raise ValueError
            model["init_args"]["num_classes"] = num_classes
            self.model = instantiate_class((), model)
        elif isinstance(model, XvecTDNN):
            self.model = model
        else:
            raise TypeError

        self.num_classes = self.model.num_classes
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


