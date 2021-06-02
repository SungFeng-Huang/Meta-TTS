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
from lightning.system import System, FewShotSystem
from lightning.collate import get_meta_collate, get_multi_collate, get_single_collate
from lightning.utils import seed_all, EpisodicInfiniteWrapper


class BaselineSystem(FewShotSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # All of the settings below are for few-shot validation

        self.test_ways      = self.train_config["meta"]["ways"]
        self.test_shots     = self.train_config["meta"]["shots"]
        self.test_queries   = 1
        # self.test_queries   = self.train_config["meta"]["queries"]
        self.test_adaptation_steps = 100

        meta_config = self.train_config["meta"]
        self.adaptation_steps   = meta_config.get("adaptation_steps", LightningMAML.adaptation_steps)
        self.adaptation_lr      = meta_config.get("adaptation_lr", LightningMAML.adaptation_lr)
        self.data_parallel      = meta_config.get("data_parallel", False)

        self.encoder = self.model.encoder
        self.learner = torch.nn.ModuleDict({
            'variance_adaptor'  : self.model.variance_adaptor,
            'decoder'           : self.model.decoder,
            'mel_linear'        : self.model.mel_linear,
            'postnet'           : self.model.postnet,
            'speaker_emb'       : self.model.speaker_emb,
        })

        # if self.data_parallel and torch.cuda.device_count() > 1:
            # self.encoder = torch.nn.DataParallel(self.encoder)
        self.learner = l2l.algorithms.MAML(self.learner, lr=self.adaptation_lr)

    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None):
        if learner is None:
            learner = self.learner.clone()
            learner.train()

        sup_batch = batch[0][0][0]

        # Adapt the classifier
        for step in range(adaptation_steps):
            preds = self.forward_learner(
                learner, *(sup_batch[2:])
            )
            train_error = self.loss_func(sup_batch, preds)
            learner.adapt(train_error[0], first_order=(not self.trainer.training), allow_unused=False, allow_nograd=True)
        return learner

    def meta_learn(self, batch, batch_idx, ways, shots, queries):
        # self.encoder.train()
        learner = self.adapt(batch, self.adaptation_steps)

        # Evaluating the adapted model
        qry_batch = batch[0][1][0]
        predictions = self.forward_learner(
            learner, *(qry_batch[2:])
        )
        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx):
        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]

        val_loss, predictions = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries
        )

        # Synthesis one sample and log to CometLogger
        if batch_idx == 0 and self.local_rank == 0:
            metadata = {'stage': "Validation", 'sup_ids': sup_ids}
            qry_batch = batch[0][1][0]
            fig, wav_reconstruction, wav_prediction, basename = self.synth_one_sample(qry_batch, predictions)
            step = self.global_step+1
            self.log_figure(
                f"Validation/step_{step}_{basename}", fig
            )
            self.log_audio(
                f"Validation/step_{step}_{basename}_reconstructed", wav_reconstruction,
                metadata=metadata
            )
            self.log_audio(
                f"Validation/step_{step}_{basename}_synthesized", wav_prediction,
                metadata=metadata
            )

        # Log loss for each sample to csv files
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        self.log_csv(
            val_loss,
            log_dir=os.path.join(self.log_dir, "val_csv"),
            file_name=f"{self.val_SQids2Tid[SQids]}.csv",
            step=self.global_step
        )

        # Log metrics to CometLogger
        tblog_dict = self.loss2dict(val_loss)
        self.log_dict(
            {f"val_{k}":v for k, v in tblog_dict.items()}, sync_dist=True,
        )
        return tblog_dict

    def validation_epoch_end(self, val_outputs=None):
        """Log hp_metric to tensorboard for hparams selection."""
        if self.global_step > 0:
            tblog_dict = merge_dicts(val_outputs)
            loss = self.dict2loss(tblog_dict)

            # Log total loss to log.txt and print to stdout
            message = f"Validation Step {self.global_step+1}, "
            message += self.loss2str(loss)
            self.log_text(message)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        ways, shots, queries = self.test_ways, self.test_shots, self.test_queries
        test_adaptation_steps = self.test_adaptation_steps

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        sup_ids = sup_batch[0]
        qry_ids = qry_batch[0]
        metadata = {'stage': "Testing", 'sup_ids': sup_ids}
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"

        # Evaluating the initial model
        predictions = self.forward_learner(
            self.learner, *(qry_batch[2:]),
        )
        valid_error = self.loss_func(qry_batch, predictions)

        if hasattr(self, 'test_global_step'):
            global_step = self.test_global_step
        else:
            global_step = self.global_step
        self.log_csv(
            valid_error,
            log_dir=os.path.join(self.result_dir, "test_csv", f"step_{global_step}"),
            file_name=f"{self.test_SQids2Tid[SQids]}.csv",
            step=0
        )

        self.recon_samples(
            qry_batch, predictions,
            tag=f"Testing/{self.test_SQids2Tid[SQids]}",
            log_dir=self.result_dir,
        )

        # synth_samples & save & log
        predictions = self.forward_learner(
            self.learner, *(qry_batch[2:6]),
        )
        self.synth_samples(
            qry_batch, predictions,
            tag=f"Testing/{self.test_SQids2Tid[SQids]}",
            name=f"step_{global_step}-FTstep_0",
            log_dir=self.result_dir,
        )

        # Adapt
        assert test_adaptation_steps % self.adaptation_steps == 0
        learner = None
        for ft_step in range(self.adaptation_steps, test_adaptation_steps+1, self.adaptation_steps):
            learner = self.adapt(batch, self.adaptation_steps, learner=learner)

            # Evaluating the adapted model
            predictions = self.forward_learner(
                learner, *(qry_batch[2:]),
            )
            valid_error = self.loss_func(qry_batch, predictions)

            self.log_csv(
                valid_error,
                log_dir=os.path.join(self.result_dir, "test_csv", f"step_{global_step}"),
                file_name=f"{self.test_SQids2Tid[SQids]}.csv",
                step=ft_step
            )

            if ft_step in [5, 10, 20, 50, 100]:
                # synth_samples & save & log
                predictions = self.forward_learner(
                    learner, *(qry_batch[2:6]),
                )
                self.synth_samples(
                    qry_batch, predictions,
                    tag=f"Testing/{self.test_SQids2Tid[SQids]}",
                    name=f"step_{global_step}-FTstep_{ft_step}",
                    log_dir=self.result_dir,
                )

    def train_dataloader(self):
        if not isinstance(self.train_dataset, EpisodicInfiniteWrapper):
            meta_batch_size = self.train_config["meta"]["meta_batch_size"]
            train_ways      = self.train_config["meta"]["ways"]
            train_shots     = self.train_config["meta"]["shots"]
            train_queries   = self.train_config["meta"]["queries"]

            assert torch.cuda.device_count() == 1
            batch_size = train_ways * (train_shots + train_queries) * meta_batch_size
            val_step = self.train_config["step"]["val_step"]
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset, val_step*batch_size)

        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=get_single_collate(False),
            num_workers=8,
        )
        return self.train_loader

    def val_dataloader(self):
        val_task_dataset = self._few_shot_task_dataset(
            self.val_dataset,
            self.test_ways,
            self.test_shots,
            self.test_queries,
            task_per_speaker=8,
        )

        # Fix random seed for validation set. Don't use pl.seed_everything(43,
        # True) if don't want to affect training seed. Use my seed_all instead.
        pl.seed_everything(43, True)
        # with seed_all(43):
        self.val_SQids2Tid = self.prefetch_tasks(val_task_dataset, 'val', self.log_dir)

        # DataLoader
        self.val_loader = DataLoader(
            val_task_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
            num_workers=8,
        )
        return self.val_loader

    def test_dataloader(self):
        test_task_dataset = self._few_shot_task_dataset(
            self.test_dataset,
            self.test_ways,
            self.test_shots,
            self.test_queries,
            task_per_speaker=16,
        )

        # Fix random seed for testing set. Don't use pl.seed_everything(43,
        # True) if don't want to affect training seed. Use my seed_all instead.
        with seed_all(43):
            self.test_SQids2Tid = self.prefetch_tasks(test_task_dataset, 'test', self.result_dir)

        # DataLoader
        self.test_loader = DataLoader(
            test_task_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
            num_workers=8,
        )
        return self.test_loader

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # if hasattr(self, 'val_SQids2vid'):
            # checkpoint["val_SQids2Tid"] = self.val_SQids2vid
        # elif hasattr(self, 'val_SQids2tid'):
            # checkpoint["val_SQids2Tid"] = self.val_SQids2tid
        # elif hasattr(self, 'val_SQids2Tid'):
            # checkpoint["val_SQids2Tid"] = self.val_SQids2Tid
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        # self.val_SQids2Tid = checkpoint["val_SQids2Tid"]
        self.test_global_step = checkpoint["global_step"]

    def forward_learner(
        self, learner,
        speakers, texts,
        src_lens, max_src_len,
        mels=None, mel_lens=None, max_mel_len=None,
        p_targets=None, e_targets=None, d_targets=None,
        p_control=1.0, e_control=1.0, d_control=1.0,
    ):
        encoder = self.model.encoder
        variance_adaptor = self.model.variance_adaptor
        decoder = self.model.decoder
        mel_linear = self.model.mel_linear
        postnet = self.model.postnet
        if 'variance_adaptor' in learner.module:
            variance_adaptor = learner.module['variance_adaptor']
        if 'decoder' in learner.module:
            decoder = learner.module['decoder']
        if 'mel_linear' in learner.module:
            mel_linear = learner.module['mel_linear']
        if 'postnet' in learner.module:
            postnet = learner.module['postnet']
        if learner.module['speaker_emb'] is not None:
            speaker_emb = learner.module['speaker_emb']

        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        output = encoder(texts, src_masks)

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        )

        if learner.module['speaker_emb'] is not None:
            output = output + speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_masks,
        ) = variance_adaptor(
            output,
            src_masks, mel_masks, max_mel_len,
            p_targets, e_targets, d_targets, p_control, e_control, d_control,
        )

        if learner.module['speaker_emb'] is not None:
            output = output + speaker_emb(speakers).unsqueeze(1).expand(
                -1, max(mel_lens), -1
            )

        output, mel_masks = decoder(output, mel_masks)
        output = mel_linear(output)

        postnet_output = postnet(output) + output

        return (
            output, postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens,
        )


class BaselineEmb1VADecSystem(BaselineSystem):
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
