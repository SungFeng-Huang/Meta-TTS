#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l
from scipy.io import wavfile
from tqdm import tqdm

from torch.utils.data import DataLoader
from pytorch_lightning.loggers.base import merge_dicts
from learn2learn.algorithms.lightning import LightningMAML
from learn2learn.utils.lightning import EpisodicBatcher

from model import FastSpeech2Loss, FastSpeech2
from utils.tools import get_mask_from_lengths, expand, plot_mel
from lightning.baseline import BaselineSystem
from lightning.collate import get_meta_collate, get_multi_collate, get_single_collate
from lightning.utils import seed_all, EpisodicInfiniteWrapper


class BaselineEmbDecSystem(BaselineSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = self.model.encoder
        del self.learner

        self.variance_adaptor = self.model.variance_adaptor
        self.learner = torch.nn.ModuleDict({
            'decoder'           : self.model.decoder,
            'mel_linear'        : self.model.mel_linear,
            'postnet'           : self.model.postnet,
            'speaker_emb'       : self.model.speaker_emb,
        })

        self.learner = l2l.algorithms.MAML(self.learner, lr=self.adaptation_lr)


class BaselineEmb1DecSystem(BaselineEmbDecSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    1 embedding for all speakers.
    """

    def __init__(
        self,
        model=None,
        optimizer=None,
        loss_func=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        scheduler=None,
        configs=None,
        vocoder=None,
        log_dir=None,
        result_dir=None,
     ):
        preprocess_config, model_config, train_config = configs

        if model is None:
            model = FastSpeech2(preprocess_config, model_config)
        if loss_func is None:
            loss_func = FastSpeech2Loss(preprocess_config, model_config)

        if model_config["multi_speaker"]:
            del model.speaker_emb
            model.speaker_emb = torch.nn.Embedding(
                1,
                model_config["transformer"]["encoder_hidden"],
            )
        
        del optimizer
        del scheduler
        optimizer = None
        scheduler = None

        super().__init__(
            model, optimizer, loss_func,
            train_dataset, val_dataset, test_dataset,
            scheduler, configs, vocoder,
            log_dir, result_dir
        )

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        with torch.no_grad():
            batch[2].zero_()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        with torch.no_grad():
            batch[0][0][0][2].zero_()
            batch[0][1][0][2].zero_()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        with torch.no_grad():
            batch[0][0][0][2].zero_()
            batch[0][1][0][2].zero_()
