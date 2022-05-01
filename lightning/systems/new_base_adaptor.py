#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pandas as pd
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
        self.learner = MAML(self.model, lr=adaptation_lr)
        # self.learner.inner_freeze()
        # for module in self.algorithm_config["adapt"]["modules"]:
            # self.learner.inner_unfreeze(module)

        self.adaptation_steps      = self.algorithm_config["adapt"]["train"]["steps"]
        self.test_adaptation_steps = self.algorithm_config["adapt"]["test"]["steps"]
        assert self.test_adaptation_steps % self.adaptation_steps == 0

        self.d_target = False
        self.reference_prosody = True

    def on_train_epoch_start(self):
        self.train_loss_dicts = []

    def on_train_batch_start(self, batch, batch_idx):
        """
        # global_step not yet +1 before train batch start
        # batch_idx start from 0 for each epoch
        # at the first epoch, global_step == batch_idx
        """
        # self.print("start", self.global_step, batch_idx)
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        # global_step already +1 after train batch end
        # at the first epoch, global_step == batch_idx + 1
        """
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.print({k: v.item() for k, v in
                        self.trainer.callback_metrics.items()})
            key_map = {
                "Train/Total Loss": "total_loss",
                "Train/Mel Loss": "mel_loss",
                "Train/Mel-Postnet Loss": "_mel_loss",
                "Train/Duration Loss": "d_loss",
                "Train/Pitch Loss": "p_loss",
                "Train/Energy Loss": "e_loss",
            }
            loss_dict = {"step": self.global_step}
            for k in key_map:
                loss_dict[key_map[k]] = self.trainer.callback_metrics[k].item()
            self.train_loss_dicts.append(loss_dict)
            df = pd.DataFrame(self.train_loss_dicts).set_index("step")
            if batch_idx + 1 == self.trainer.log_every_n_steps:
                self.print(df.to_string(header=True, index=True))
            else:
                self.print(df.to_string(header=True, index=True).split('\n')[-1])

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self._on_meta_batch_start(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.trainer.state.fn == "fit":
            # self.trainer.state.fn == TrainerFn.FITTING == "fit"
            return self._val_step_for_fit(batch, batch_idx, dataloader_idx)

    def on_validation_end(self):
        if self.trainer.state.fn == "fit":
            # self.trainer.state.fn == TrainerFn.FITTING == "fit"
            return self._on_val_end_for_fit()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        all_outputs = []
        qry_batch = batch[0][1][0]
        if self.algorithm_config["adapt"]["test"].get("1-shot", False):
            task = Task(sup_data=batch[0][0][0], batch_size=1, shuffle=False)
            for sup_batch in iter(task):
                mini_batch = [([sup_batch], [qry_batch])]
                outputs = self._test_step(mini_batch, batch_idx)
                all_outputs.append(outputs)
        else:
            outputs = self._test_step(batch, batch_idx, dataloader_idx)
            all_outputs.append(outputs)
        torch.distributed.barrier()

        return all_outputs

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2, "sup + qry"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 12, "data with 12 elements"

    def _val_step_for_fit(self, batch, batch_idx, dataloader_idx=0):
        """ Adapted forwarding.

        Function:
            meta_learn(): Defined in `lightning.systems.base_adaptor.BaseAdaptorSystem`
        """
        val_loss, predictions = self.meta_learn(batch, batch_idx, train=False)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=1)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}

    def _on_val_end_for_fit(self):
        if self.global_step > 0:
            self.print(f"Step {self.global_step} - Val losses")
            loss_dicts = []
            for i in range(len(self.trainer.datamodule.val_datasets)):
                key_map = {
                    f"Val/Total Loss/dataloader_idx_{i}": "total_loss",
                    f"Val/Mel Loss/dataloader_idx_{i}": "mel_loss",
                    f"Val/Mel-Postnet Loss/dataloader_idx_{i}": "_mel_loss",
                    f"Val/Duration Loss/dataloader_idx_{i}": "d_loss",
                    f"Val/Pitch Loss/dataloader_idx_{i}": "p_loss",
                    f"Val/Energy Loss/dataloader_idx_{i}": "e_loss",
                }
                loss_dict = {"dataloader_idx": int(i)}
                for k in key_map:
                    loss_dict[key_map[k]] = self.trainer.callback_metrics[k].item()
                loss_dicts.append(loss_dict)

            df = pd.DataFrame(loss_dicts).set_index("dataloader_idx")
            self.print(df.to_string(header=True, index=True)+'\n')

    def _test_step(self, batch, batch_idx, dataloader_idx):
        outputs = {}

        assert "saving_steps" in self.algorithm_config["adapt"]["test"]
        saving_steps = self.algorithm_config["adapt"]["test"]["saving_steps"]

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        reference_prosody = qry_batch[9].unsqueeze(2) if self.reference_prosody else None
        outputs['_batch'] = qry_batch

        # Evaluating the initial model
        predictions = self.learner(sup_batch[2], *qry_batch[3:],
                                   average_spk_emb=True,
                                   reference_prosody=reference_prosody)
        valid_error = self.loss_func(qry_batch, predictions)
        outputs[f"step_0"] = {"recon": {"losses": valid_error, "output": predictions}}

        # synth_samples & save & log
        if self.d_target:
            predictions = self.learner(sup_batch[2], *qry_batch[3:9],
                                       d_targets=qry_batch[11],
                                       average_spk_emb=True,
                                       reference_prosody=reference_prosody)
        else:
            predictions = self.learner(sup_batch[2], *qry_batch[3:6],
                                       average_spk_emb=True,
                                       reference_prosody=reference_prosody)
        outputs[f"step_0"].update({"synth": {"output": predictions}})

        # Adapt
        learner = None
        for ft_step in range(self.adaptation_steps, self.test_adaptation_steps+1, self.adaptation_steps):
            learner = self.adapt(batch, self.adaptation_steps, learner=learner, train=False)

            # Evaluating the adapted model
            learner.eval()
            if self.d_target:
                predictions = learner(sup_batch[2], *qry_batch[3:9],
                                      d_targets=qry_batch[11],
                                      average_spk_emb=True,
                                      reference_prosody=reference_prosody)
            else:
                predictions = learner(sup_batch[2], *qry_batch[3:],
                                      average_spk_emb=True,
                                      reference_prosody=reference_prosody)
            valid_error = self.loss_func(qry_batch, predictions)
            outputs[f"step_{ft_step}"] = {"recon": {"losses": valid_error, "output": predictions}}

            if ft_step in saving_steps:
                # synth_samples & save & log
                if self.d_target:
                    predictions = learner(sup_batch[2], *qry_batch[3:9],
                                          d_targets=qry_batch[11],
                                          average_spk_emb=True,
                                          reference_prosody=reference_prosody)
                else:
                    predictions = learner(sup_batch[2], *qry_batch[3:6],
                                          average_spk_emb=True,
                                          reference_prosody=reference_prosody)
                outputs[f"step_{ft_step}"].update({"synth": {"output": predictions}})
        del learner

        return outputs


    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, train=True):
        if learner is None:
            learner = self.learner.clone()
            learner.inner_freeze()
            for module in self.algorithm_config["adapt"]["modules"]:
                learner.inner_unfreeze(module)
        learner.train()

        # Adapt the classifier
        sup_batch = batch[0][0][0]
        reference_prosody = sup_batch[9].unsqueeze(2) if self.reference_prosody else None
        first_order = not train
        for step in range(adaptation_steps):
            preds = learner(*sup_batch[2:],
                            reference_prosody=reference_prosody)
            train_error = self.loss_func(sup_batch, preds)
            learner.adapt_(train_error[0],
                           first_order=first_order,
                           allow_unused=False,
                           allow_nograd=True)
        return learner

    def meta_learn(self, batch, batch_idx, train=True):
        learner = self.adapt(batch, min(self.adaptation_steps, self.test_adaptation_steps),
                             train=train)
        learner.eval()

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        reference_prosody = qry_batch[9].unsqueeze(2) if self.reference_prosody else None

        # Evaluating the adapted model
        # Use speaker embedding of support set
        if self.d_target:
            predictions = learner(sup_batch[2], *qry_batch[3:9],
                                  d_targets=qry_batch[11],
                                  average_spk_emb=True,
                                  reference_prosody=reference_prosody)
        else:
            predictions = learner(sup_batch[2], *qry_batch[3:],
                                  average_spk_emb=True,
                                  reference_prosody=reference_prosody)
        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions

