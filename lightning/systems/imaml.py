#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l

from typing import List, Callable
from torch import Tensor
from torch.nn import Module
from learn2learn.algorithms.lightning import LightningMAML
from hypergrad import update_tensor_grads

from utils.tools import get_mask_from_lengths
from lightning.utils import loss2dict
from lightning.systems.base_adaptor import BaseAdaptorSystem
from lightning.systems.utils import Task, CG, MAML


class IMAMLSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # NOTE: `self.model.encoder` and `self.learner.encoder` are pointing to
        # the same variable, they are not two variables with the same values.
        sup_batch, qry_batch, ref_phn_feats = batch[0]
        # ref_phn_feats = batch[0][2]
        self.model.encoder.src_word_emb._parameters['weight'] = \
            self.model.phn_emb_generator.get_new_embedding(ref_phn_feats).clone()
        # del batch[0][2]
        # return batch
        return [(sup_batch, qry_batch)]

    def bias_reg_f(self, learner):
        # l2 biased regularization
        return sum([((b - p) ** 2).sum()
                    for b, p in zip(
                        self.learner.parameters(), learner.parameters()
                    )])

    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, task=None, train=True):
        """ Regularized adapt for iMAML. """
        if learner is None:
            learner = self.learner.clone()
            learner.train()

        if task is None:
            sup_data = batch[0][0][0]
            qry_data = batch[0][1][0]
            task = Task(sup_data=sup_data,
                        qry_data=qry_data,
                        batch_size=self.algorithm_config["adapt"]["imaml"]["batch_size"])

        # Adapt the classifier
        first_order = True
        reg_param = self.algorithm_config["adapt"]["imaml"]["reg_param"]
        for step in range(adaptation_steps):
            mini_batch = task.next_batch()
            preds = self.forward_learner(learner, *mini_batch[2:])
            train_error = self.loss_func(mini_batch, preds)
            loss = (train_error[0] + 0.5 * reg_param * self.bias_reg_f(learner))
            learner.adapt_(loss, first_order=first_order, allow_unused=False, allow_nograd=True)
        return learner, task

    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, train=True):
        sup_data = batch[0][0][0]
        qry_data = batch[0][1][0]

        # self.gpu_stats(f"Step {self.global_step}: Before inner update")
        learner, task = self.adapt(batch, self.adaptation_steps, train=train)
        # self.gpu_stats(f"Step {self.global_step}: After inner update")
        stochastic = self.algorithm_config["adapt"]["imaml"]["stochastic"]
        reg_param = self.algorithm_config["adapt"]["imaml"]["reg_param"]

        def fp_map(learner: MAML, hmodel: Module) -> List[Tensor]:
            """ Required by CG. Adaptation mapping function.
            Input: learner module, hyper-model.
            Output: updated learner params.
            """
            if stochastic:
                mini_batch = task.next_batch()
            else:
                mini_batch = sup_data
            preds = self.forward_learner(learner, *mini_batch[2:])
            losses = self.loss_func(mini_batch, preds)
            # biased regularized loss where the bias are the meta-parameters in hparams
            loss = (losses[0] + 0.5 * reg_param * self.bias_reg_f(learner))
            # new_learner = learner.copy()
            # new_learner.adapt_(loss, first_order=False, allow_unused=False, allow_nograd=True)
            new_learner = learner.adapt(loss, first_order=False, allow_unused=False, allow_nograd=True)
            # return list(new_learner.parameters())
            return [p for p in new_learner.module.parameters() if p.requires_grad]

        def outer_loss(learner: MAML, hmodel: Module) -> Tensor:
            """ Required by CG. Return query loss for meta-update.
            Input: learner module, hyper-model.
            Output: query loss.
            """
            mini_batch = qry_data
            preds = self.forward_learner(learner, sup_data[2], *mini_batch[3:], average_spk_emb=True)
            losses = self.loss_func(mini_batch, preds)
            return losses[0], losses, preds
            
        if train:
            task.reset_iterator()
            grads, valid_error, predictions = CG(learner, self.model,
                                                 K=self.algorithm_config["adapt"]["imaml"]["K"],
                                                 fp_map=fp_map,
                                                 outer_loss=outer_loss,
                                                 set_grad=False,
                                                 stochastic=True)
            # self.gpu_stats(f"Step {self.global_step}: After CG")

            with torch.no_grad():
                total_norm = torch.norm(torch.stack([torch.norm(g, 2.0).to(self.device) for g in grads]), 2.0)
                max_norm = self.train_config["optimizer"]["grad_clip_thresh"]
                clip_coef = max_norm / (total_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0).to(self.device)
                for g in grads:
                    g.mul_(clip_coef_clamped)
            grads = [self.trainer.training_type_plugin.reduce(g) for g in grads]

            opt = self.optimizers()
            opt.zero_grad()

            hparams = [p for p in self.model.parameters() if p.requires_grad]
            update_tensor_grads(hparams, grads)
            opt.step()

            sch = self.lr_schedulers()
            sch.step()

            del task

        else:
            with torch.no_grad():
                _, valid_error, predictions = outer_loss(learner, self.model)

        return valid_error, predictions


    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.

        Function:
            common_step(): Defined in `lightning.systems.system.System`
        """
        train_loss, predictions = self.meta_learn(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0], 'losses': train_loss, 'output': predictions, '_batch': qry_batch}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)

    def validation_step(self, batch, batch_idx):
        """ Adapted forwarding.

        Function:
            meta_learn(): Defined in `lightning.systems.base_adaptor.BaseAdaptorSystem`
        """
        val_loss, predictions = self.meta_learn(batch, batch_idx)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}

    def test_step(self, batch, batch_idx):
        outputs = {}

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
        task = None
        for ft_step in range(self.adaptation_steps, self.test_adaptation_steps+1, self.adaptation_steps):
            learner, task = self.adapt(batch, self.adaptation_steps, learner=learner, task=task, train=False)

            # Evaluating the adapted model
            predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:], average_spk_emb=True)
            valid_error = self.loss_func(qry_batch, predictions)
            outputs[f"step_{ft_step}"] = {"recon": {"losses": valid_error, "output": predictions}}

            if ft_step in [5, 10, 20, 50, 100]:
                # synth_samples & save & log
                predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
                outputs[f"step_{ft_step}"].update({"synth": {"output": predictions}})
        del learner
        del task

        return outputs

    def gpu_stats(self, prefix=""):
        # t = torch.cuda.get_device_properties(self.local_rank).total_memory
        # r = torch.cuda.memory_reserved(self.local_rank)
        # a = torch.cuda.memory_allocated(self.local_rank)
        # f = r-a  # free inside reserved
        print(prefix
              + '\n' + torch.cuda.memory_summary(self.device, True))
              # + f"\n{'-'*21}"
              # + f"\n{'GPU stats': ^21}"
              # + f"\n{'='*21}"
              # + f"\nTotal: {t}"
              # + f"\nReserved(Cached): {r}"
              # + f"\nAllocated: {a}"
              # + f"\nFree: {f}"
              # + f"\n{'-'*21}\n")

