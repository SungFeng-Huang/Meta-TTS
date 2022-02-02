#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l
from tqdm import tqdm
import matplotlib.pyplot as plt

from tqdm import tqdm
from resemblyzer import VoiceEncoder
# from learn2learn.algorithms import MAML

from .utils import MAML, CodebookAnalyzer
from utils.tools import get_mask_from_lengths
from lightning.systems.utils import Task
from lightning.systems.system import System
from lightning.utils import loss2dict


class BaseAdaptorSystem(System):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # All of the settings below are for few-shot validation
        self.adaptation_lr = self.algorithm_config["adapt"]["task"]["lr"]
        self.adaptation_class = self.algorithm_config["adapt"]["class"]
        # self.learner = MAML(
        #     torch.nn.ModuleDict({
        #         k: getattr(self.model, k) for k in self.algorithm_config["adapt"]["modules"]
        #     }), lr=adaptation_lr
        # )

        self.adaptation_steps      = self.algorithm_config["adapt"]["train"]["steps"]
        self.test_adaptation_steps = self.algorithm_config["adapt"]["test"]["steps"]
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)
        assert self.adaptation_steps == 0 or self.test_adaptation_steps % self.adaptation_steps == 0

    def build_learner(self, *args, **kwargs):
        return None

    def get_matching(self, *args, **kwargs):
        return None
    
    def forward_learner(
        self, learner, speaker_args, texts, src_lens, max_src_len,
        mels=None, mel_lens=None, max_mel_len=None,
        p_targets=None, e_targets=None, d_targets=None,
        p_control=1.0, e_control=1.0, d_control=1.0,
        average_spk_emb=False,
    ):
        _get_module = lambda name: getattr(learner.module, name, getattr(self.model, name, None))
        embedding        = _get_module('embedding')
        encoder          = _get_module('encoder')
        variance_adaptor = _get_module('variance_adaptor')
        decoder          = _get_module('decoder')
        mel_linear       = _get_module('mel_linear')
        postnet          = _get_module('postnet')
        speaker_emb      = _get_module('speaker_emb')

        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        if torch.isnan(texts).any():
            print(texts)
            print("text NaN!")
            print("Num of NaN: ", torch.isnan(texts).sum().item())
        emb_texts = embedding(texts)
        if torch.isnan(emb_texts).any():
            print(emb_texts)
            print("emb text NaN!")
            print("Num of NaN: ", torch.isnan(emb_texts).sum().item())
        output = encoder(emb_texts, src_masks)
        if torch.isnan(output).any():
            print("encoder NaN!")

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        )

        if speaker_emb is not None:
            if torch.isnan(speaker_args[0]).any():
                print("ref mel NaN!")
                print(speaker_args[0].shape)
                print("Num of NaN: ", torch.isnan(speaker_args[0]).sum().item())
            spk_emb = speaker_emb(speaker_args)
            if torch.isnan(spk_emb).any():
                print("spk emb NaN!")
                print(spk_emb.shape)
                print("Num of NaN: ", torch.isnan(spk_emb).sum().item())
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            output += spk_emb.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output, p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_masks,
        ) = variance_adaptor(
            output, src_masks, mel_masks, max_mel_len,
            p_targets, e_targets, d_targets, p_control, e_control, d_control,
        )
        if torch.isnan(output).any():
            print("variance adaptor NaN!")

        if speaker_emb is not None:
            spk_emb = speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            if max_mel_len is None:  # inference stage
                max_mel_len = max(mel_lens)
            output += spk_emb.unsqueeze(1).expand(-1, max_mel_len, -1)

        output, mel_masks = decoder(output, mel_masks)
        if torch.isnan(output).any():
            print("decoder NaN!")
        output = mel_linear(output)
        if torch.isnan(output).any():
            print("mel linear NaN!")

        tmp = postnet(output)
        if torch.isnan(tmp).any():
            print("mel postnet NaN!")
        postnet_output = tmp + output

        return (
            output, postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens,
        )

    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, task=None, train=True):
        if learner is None:
            learner = self.learner.clone()
            learner.train()

        # Adapt the classifier
        first_order = not train
        for step in range(adaptation_steps):
            mini_batch = task.next_batch()
            preds = self.forward_learner(learner, *mini_batch[2:])
            train_error = self.loss_func(mini_batch, preds)
            learner.adapt_(train_error[0], first_order=first_order, allow_unused=False, allow_nograd=True)
        return learner

    def meta_learn(self, batch, batch_idx, train=True):
        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        task = Task(sup_data=sup_batch,
                    qry_data=qry_batch,
                    batch_size=self.algorithm_config["adapt"]["imaml"]["batch_size"]) # The batch size is borrowed from the imaml config.


        learner = self.adapt(batch, min(self.adaptation_steps, self.test_adaptation_steps), task=task, train=train)

        # Evaluating the adapted model
        # Use speaker embedding of support set
        # predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:], average_spk_emb=True)
        predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2 or len(batch[0]) == 4, "sup + qry (+ ref_phn_feats + lang_id)"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 12, "data with 12 elements"

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        outputs = {}

        learner = self.build_learner(batch)
        matching = self.get_matching(batch)
        self.codebook_analyzer.visualize_matching(batch_idx, matching)

        sup_batch, qry_batch, _, _ = batch[0]
        batch = [(sup_batch, qry_batch)]
        
        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        outputs['_batch'] = qry_batch

        # Evaluating the initial model
        # predictions = self.forward_learner(self.learner, sup_batch[2], *qry_batch[3:], average_spk_emb=True)
        learner.eval()
        self.model.eval()
        with torch.no_grad():
            predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
            valid_error = self.loss_func(qry_batch, predictions)
            outputs[f"step_0"] = {"recon": {"losses": valid_error, "output": predictions}}
      
            # synth_samples & save & log
            # predictions = self.forward_learner(self.learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
            predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
            outputs[f"step_0"].update({"synth": {"output": predictions}})
        learner.train()
        self.model.train()

        # Adapt
        # learner = None
        learner = learner.clone()
        task = Task(sup_data=sup_batch,
                    qry_data=qry_batch,
                    batch_size=self.algorithm_config["adapt"]["test"]["batch_size"])

        # evaluate training data
        fit_batch = task.next_batch()
        outputs['_batch_fit'] = fit_batch

        learner.eval()
        self.model.eval()
        with torch.no_grad():
            predictions = self.forward_learner(learner, *fit_batch[2:], average_spk_emb=True)
            outputs[f"step_0"].update({"recon-fit": {"output": predictions}})
        learner.train()
        self.model.train()

        ft_steps = [100, 250, 500, 750, 1000]
        self.test_adaptation_steps = max(ft_steps)
        
        for ft_step in tqdm(range(self.adaptation_steps, self.test_adaptation_steps+1, self.adaptation_steps)):
            learner = self.adapt(batch, self.adaptation_steps, learner=learner, task=task, train=False)
            
            learner.eval()
            self.model.eval()
            with torch.no_grad():
                # Evaluating the adapted model
                # predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:], average_spk_emb=True)
                predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
                valid_error = self.loss_func(qry_batch, predictions)
                outputs[f"step_{ft_step}"] = {"recon": {"losses": valid_error}}

                if ft_step in ft_steps:
                    # synth_samples & save & log
                    # predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
                    fit_preds = self.forward_learner(learner, *fit_batch[2:6], average_spk_emb=True)
                    outputs[f"step_{ft_step}"].update({"synth-fit": {"output": fit_preds}})
                    predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
                    outputs[f"step_{ft_step}"].update({"synth": {"output": predictions}})
            learner.train()
            self.model.train()
        del learner

        return outputs
