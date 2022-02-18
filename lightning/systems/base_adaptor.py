#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l

from tqdm import tqdm
from resemblyzer import VoiceEncoder
# from learn2learn.algorithms import MAML

from .utils import MAML
from utils.tools import get_mask_from_lengths
from lightning.systems.system import System
from lightning.systems.utils import Task
from lightning.utils import loss2dict


class BaseAdaptorSystem(System):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # All of the settings below are for few-shot validation
        adaptation_lr = self.algorithm_config["adapt"]["task"]["lr"]
        # self.adaptation_class = self.algorithm_config["adapt"]["class"]
        self.learner = MAML(
            torch.nn.ModuleDict({
                k: getattr(self.model, k) for k in self.algorithm_config["adapt"]["modules"]
            }), lr=adaptation_lr
        )

        self.adaptation_steps      = self.algorithm_config["adapt"]["train"]["steps"]
        self.test_adaptation_steps = self.algorithm_config["adapt"]["test"]["steps"]
        assert self.test_adaptation_steps % self.adaptation_steps == 0

    def forward_learner(
        self, learner, speaker_args, texts, src_lens, max_src_len,
        mels=None, mel_lens=None, max_mel_len=None,
        p_targets=None, e_targets=None, d_targets=None,
        p_control=1.0, e_control=1.0, d_control=1.0,
        average_spk_emb=False,
    ):
        _get_module = lambda name: getattr(learner.module, name, getattr(self.model, name, None))
        encoder          = _get_module('encoder')
        variance_adaptor = _get_module('variance_adaptor')
        decoder          = _get_module('decoder')
        mel_linear       = _get_module('mel_linear')
        postnet          = _get_module('postnet')
        speaker_emb      = _get_module('speaker_emb')

        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        output = encoder(texts, src_masks)

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        )

        if speaker_emb is not None:
            spk_emb = speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)

        if speaker_emb is not None:
            output += spk_emb.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output, p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_masks,
        ) = variance_adaptor(
            output, src_masks, mel_masks, max_mel_len,
            p_targets, e_targets, d_targets, p_control, e_control, d_control,
        )

        if speaker_emb is not None:
            # spk_emb = speaker_emb(speaker_args)
            # if average_spk_emb:
                # spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            output += spk_emb.unsqueeze(1).expand(-1, max(mel_lens), -1)

        output, mel_masks = decoder(output, mel_masks)
        output = mel_linear(output)

        postnet_output = postnet(output) + output

        return (
            output, postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens,
        )

    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, train=True):
        if learner is None:
            learner = self.learner.clone()
            learner.train()

        # Adapt the classifier
        sup_batch = batch[0][0][0]
        first_order = not train
        for step in range(adaptation_steps):
            preds = self.forward_learner(learner, *sup_batch[2:])
            train_error = self.loss_func(sup_batch, preds)
            learner.adapt_(train_error[0], first_order=first_order, allow_unused=False, allow_nograd=True)
        return learner

    def meta_learn(self, batch, batch_idx, train=True):
        learner = self.adapt(batch, min(self.adaptation_steps, self.test_adaptation_steps), train=train)

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]

        # Evaluating the adapted model
        # Use speaker embedding of support set
        predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:], average_spk_emb=True)
        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2, "sup + qry"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 12, "data with 12 elements"

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)

    def test_step(self, batch, batch_idx):
        all_outputs = []
        qry_batch = batch[0][1][0]
        if self.algorithm_config["adapt"]["test"].get("1-shot", False):
            task = Task(sup_data=batch[0][0][0],
                        qry_data=batch[0][1][0],
                        batch_size=1,
                        shuffle=False)
            for sup_batch in iter(task):
                mini_batch = [([sup_batch], [qry_batch])]
                outputs = self._test_step(mini_batch, batch_idx)
                all_outputs.append(outputs)
        else:
            outputs = self._test_step(batch, batch_idx)
            all_outputs.append(outputs)
        torch.distributed.barrier()

        return all_outputs

    def _test_step(self, batch, batch_idx):
        outputs = {}

        saving_steps = self.algorithm_config["adapt"]["test"].get("saving_steps", [5, 10, 20, 50, 100])

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        outputs['_batch'] = qry_batch

        # Evaluating the initial model
        predictions = self.forward_learner(self.learner, sup_batch[2], *qry_batch[3:], average_spk_emb=True)
        valid_error = self.loss_func(qry_batch, predictions)
        outputs[f"step_0"] = {"recon": {"losses": valid_error, "output": predictions}}

        # synth_samples & save & log
        predictions = self.forward_learner(self.learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
        outputs[f"step_0"].update({"synth": {"output": predictions}})

        # Adapt
        learner = None
        for ft_step in range(self.adaptation_steps, self.test_adaptation_steps+1, self.adaptation_steps):
            learner = self.adapt(batch, self.adaptation_steps, learner=learner, train=False)

            # Evaluating the adapted model
            predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:], average_spk_emb=True)
            valid_error = self.loss_func(qry_batch, predictions)
            outputs[f"step_{ft_step}"] = {"recon": {"losses": valid_error, "output": predictions}}

            if ft_step in saving_steps:
                # synth_samples & save & log
                predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
                outputs[f"step_{ft_step}"].update({"synth": {"output": predictions}})
        del learner

        return outputs

