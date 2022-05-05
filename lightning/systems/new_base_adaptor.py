#!/usr/bin/env python3

import os
import json
import time
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import learn2learn as l2l

from tqdm import tqdm
from resemblyzer import VoiceEncoder
# from learn2learn.algorithms import MAML
from learn2learn.utils import detach_module

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
        """ LightningModule interface.
        Called in the training loop at the very beginning of the epoch.
        """
        self.train_loss_dicts = []

    def on_train_batch_start(self, batch, batch_idx):
        """ LightningModule interface.
        Called in the training loop before anything happens for that batch.
        """
        # self.print("start", self.global_step, batch_idx)
        pass

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    def on_train_epoch_end(self):
        """ LightningModule interface.
        Called in the training loop after the batch.
        """
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
        if (self.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0:
            self.print(df.to_string(header=True, index=True))
        else:
            self.print(df.to_string(header=True, index=True).split('\n')[-1])

    def on_validation_start(self,):
        self.mem_stats = torch.cuda.memory_stats(self.device)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ LightningModule interface.
        Called in the validation loop before anything happens for that batch.
        """
        self._on_meta_batch_start(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ LightningModule interface.
        Operates on a single batch of data from the validation set.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
        if self.trainer.state.fn == "fit":
            return self._val_step_for_fit(batch, batch_idx, dataloader_idx)
        elif self.trainer.state.fn == "validate":
            return self._val_step_for_validate(batch, batch_idx, dataloader_idx)

    def validation_epoch_end(self, outputs):
        pass

    def on_validation_end(self):
        """ LightningModule interface.
        Called at the end of validation.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
        if self.trainer.state.fn == "fit":
            return self._on_val_end_for_fit()
        elif self.trainer.state.fn == "validate":
            pass

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        """ LightningModule interface.
        Called in the test loop before anything happens for that batch.
        """
        self._on_meta_batch_start(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        all_outputs = []
        qry_batch = batch[0][1][0]
        if self.algorithm_config["adapt"]["test"].get("1-shot", False):
            task = Task(sup_data=batch[0][0][0], batch_size=1, shuffle=False)
            for sup_batch in iter(task):
                mini_batch = [([sup_batch], [qry_batch])]
                outputs = self._test_step(mini_batch, batch_idx, dataloader_idx)
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

    def clone_learner(self, inference=False):
        learner = self.learner.clone()
        # if inference:
            # detach_module(learner, keep_requires_grad=True)
        learner.inner_freeze()
        for module in self.algorithm_config["adapt"]["modules"]:
            learner.inner_unfreeze(module)
        return learner

    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adapt_steps=5, learner=None, train=True,
              inference=False):
        """ Inner adaptation or fine-tuning. """
        if learner is None:
            learner = self.learner.clone()
            learner.inner_freeze()
            for module in self.algorithm_config["adapt"]["modules"]:
                learner.inner_unfreeze(module)
        learner.train()

        # Adapt the classifier
        sup_batch = batch[0][0][0]
        ref_psd = sup_batch[9].unsqueeze(2) if self.reference_prosody else None
        first_order = not train
        for step in range(adapt_steps):
            preds = learner(*sup_batch[2:], reference_prosody=ref_psd)
            train_error = self.loss_func(sup_batch, preds)
            learner.adapt_(
                train_error[0],
                first_order=first_order, allow_unused=False, allow_nograd=True
            )
        return learner

    def meta_learn(self, batch, adapt_steps=None, learner=None, train=True,
                   synth=False, recon=False):
        """ Inner-adapt/fine-tune + recon_predictions + (outer-/eval-)loss.
        """
        if adapt_steps is None:
            adapt_steps = min(self.adaptation_steps, self.test_adaptation_steps)
        with torch.set_grad_enabled(train):
            learner = self.adapt(batch, adapt_steps, learner=learner, train=train)
        learner.eval()

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        ref_psd = qry_batch[9].unsqueeze(2) if self.reference_prosody else None

        # Evaluating the adapted model with ground_truth d,p,e (recon_preds)
        # Use speaker embedding of support set
        with torch.set_grad_enabled(train):
            predictions = learner(
                sup_batch[2], *qry_batch[3:],
                average_spk_emb=True, reference_prosody=ref_psd
            )
            valid_error = self.loss_func(qry_batch, predictions)

        if synth:
            with torch.no_grad():
                if self.d_target:
                    synth_predictions = learner(
                        sup_batch[2], *qry_batch[3:9], d_targets=qry_batch[11],
                        average_spk_emb=True, reference_prosody=ref_psd
                    )
                else:
                    synth_predictions = learner(
                        sup_batch[2], *qry_batch[3:6],
                        average_spk_emb=True, reference_prosody=ref_psd
                    )
            if recon:
                return learner, valid_error, predictions, synth_predictions

            return learner, valid_error, synth_predictions

        return learner, valid_error, predictions

    def _val_step_for_fit(self, batch, batch_idx, dataloader_idx):
        """ validation_step() during Trainer.fit() """
        _, val_loss, predictions = self.meta_learn(batch, self.adaptation_steps, train=False)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=1)
        return {
            'losses': val_loss,
            'output': predictions,
            '_batch': batch[0][1][0],
        }

    def _on_val_end_for_fit(self):
        """ on_validation_end() during Trainer.fit() """
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

    def _val_step_for_validate(self, batch, batch_idx, dataloader_idx):
        outputs = self._test_step(batch, batch_idx, dataloader_idx)
        saving_steps = self.algorithm_config["adapt"]["test"]["saving_steps"]
        st = time.time()
        for ft_step in [0] + saving_steps:
            step = f"step_{ft_step}"
            loss_dict = loss2dict(outputs[f"step_{ft_step}"]["recon"]["losses"])
            loss_dict = {f"{outputs['task_id']}/step_{ft_step}/{k}": v for k, v in
                         loss_dict.items()}
            self.log_dict(loss_dict, sync_dist=True, batch_size=1,
                          add_dataloader_idx=False)
        self.print(f"Log_dict: {time.time() - st} sec")
        # self.print(json.dumps(self._mem_stats_diff(), indent=4))
        self.print(pd.DataFrame(self._mem_stats_diff()).T
                   .to_string(header=True, index=True))
        # self.print(torch.cuda.memory_summary(self.device, True))

        st = time.time()
        torch.cuda.empty_cache()
        self.print(f"Empty cache: {time.time() - st} sec")
        self.print(pd.DataFrame(self._mem_stats_diff()).T
                   .to_string(header=True, index=True))
        return outputs

    def _test_step(self, batch, batch_idx, dataloader_idx):
        assert "saving_steps" in self.algorithm_config["adapt"]["test"]
        saving_steps = self.algorithm_config["adapt"]["test"]["saving_steps"]
        adapt_steps = [step_ - _step for _step, step_ in
                       zip([0] + saving_steps, saving_steps)]
        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        task_id = self.trainer.datamodule.val_SQids2Tid[SQids]

        outputs = {
            "_batch": batch[0][1][0],   # qry_batch
            "task_id": task_id,
        }

        # Evaluating the initial model
        st = time.time()
        # learner = self.clone_learner(inference=True)
        learner, val_loss, recon_preds, synth_preds = self.meta_learn(
            batch, adapt_steps=0, train=False, synth=True,
            recon=True,
        )
        outputs[f"step_0"] = {
            "recon": {"losses": val_loss, "output": recon_preds},
            "synth": {"output": synth_preds},
        }
        self.print(f"Adapt 0: {time.time() - st} sec")
        self.print(pd.DataFrame(self._mem_stats_diff()).T
                   .to_string(header=True, index=True))
        # self.print(torch.cuda.memory_summary(self.device, True))

        st = time.time()
        torch.cuda.empty_cache()
        self.print(f"Empty cache: {time.time() - st} sec")
        self.print(pd.DataFrame(self._mem_stats_diff()).T
                   .to_string(header=True, index=True))

        # Adapt
        for ft_step, adapt_step in zip(saving_steps, adapt_steps):
            st = time.time()
            learner, val_loss, predictions = self.meta_learn(
                batch, adapt_step, learner=learner, train=False,
                synth=True, recon=False,
            )
            outputs[f"step_{ft_step}"] = {
                "recon": {"losses": val_loss},
                "synth": {"output": predictions},
            }
            self.print(f"Adapt ~{ft_step} ({adapt_step}): {time.time() - st} sec")
            # self.print(json.dumps(self._mem_stats_diff(), indent=4))
            self.print(pd.DataFrame(self._mem_stats_diff()).T
                       .to_string(header=True, index=True))
            # self.print(torch.cuda.memory_summary(self.device, True))

            st = time.time()
            torch.cuda.empty_cache()
            self.print(f"Empty cache: {time.time() - st} sec")
            self.print(pd.DataFrame(self._mem_stats_diff()).T
                       .to_string(header=True, index=True))

        del learner

        return outputs

    def _mem_stats_diff(self):
        new_mem_stats = torch.cuda.memory_stats(self.device)
        diff = {}
        for k in self.mem_stats:
            if '.' in k:
                _ks = k.split('.', maxsplit=1)
                if _ks[0] not in diff:
                    diff[_ks[0]] = {}
                _diff = diff[_ks[0]]
                # for _k in _ks[1:-1]:
                #     if _k not in _diff:
                #         _diff[_k] = {}
                #         _diff = _diff[_k]
                if (_ks[-1].startswith("large_pool")
                        or _ks[-1].startswith("small_pool")):
                    continue
                elif _ks[-1].startswith("all."):
                    _ks[-1] = _ks[-1][4:]
                mem_diff = new_mem_stats[k] - self.mem_stats[k]
                if "_bytes" in _ks[0]:
                    _diff[_ks[-1]] = "| ".join([
                        sizeof_fmt(new_mem_stats[k]),
                        sizeof_fmt(mem_diff, sign=True)
                    ])
                else:
                    _diff[_ks[-1]] = "| ".join([
                        f"{new_mem_stats[k]}", f"{mem_diff:+}"
                    ])
            else:
                # continue
                mem_diff = new_mem_stats[k] - self.mem_stats[k]
                diff[k] = "| ".join([
                    f"{new_mem_stats[k]}", f"{mem_diff:+}"
                ])
        self.mem_stats = new_mem_stats
        return diff

def sizeof_fmt(num, suffix="B", sign=False):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            if sign:
                return f"{num:+3.1f}{unit}{suffix}"
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    if sign:
        return f"{num:+.1f}Y{suffix}"
    return f"{num:.1f}Y{suffix}"
