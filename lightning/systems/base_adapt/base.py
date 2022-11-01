#!/usr/bin/env python3

import os
import time
import torch
import pandas as pd
import pytorch_lightning as pl
# from learn2learn.algorithms import MAML
from learn2learn.utils import detach_module
from typing import Dict, Any, Optional, Union, List

# from lightning.systems.system import System
from lightning.systems.utils import MAML
from lightning.model import FastSpeech2Loss, FastSpeech2


VERBOSE = False


import yaml
def load_yaml(yaml_in: str) -> Dict[str, Any]:
    return yaml.load(open(yaml_in, 'r'), Loader=yaml.FullLoader)


class BaseAdaptorSystem(pl.LightningModule):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self,
                 preprocess_config: Union[dict, str],
                 model_config: Union[dict, str],
                 train_config: Union[dict, str, list],
                 algorithm_config: Union[dict, str],
                 log_dir=None, result_dir=None):
        super().__init__()

        if isinstance(preprocess_config, str):
            preprocess_config = load_yaml(preprocess_config)
        if isinstance(model_config, str):
            model_config = load_yaml(model_config)
        # if isinstance(train_config, str):
        #     train_config = load_yaml(train_config)

        if isinstance(train_config, str) or isinstance(train_config, dict):
            train_config = [train_config]
        _train_config = {}
        for _config in train_config:
            if isinstance(_config, str):
                _train_config.update(load_yaml(_config))
            elif isinstance(_config, dict):
                _train_config.update(_config)
        train_config = _train_config

        if isinstance(algorithm_config, str):
            algorithm_config = load_yaml(algorithm_config)

        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config
        self.save_hyperparameters()

        self.model = FastSpeech2(preprocess_config, model_config, algorithm_config)
        self.loss_func = FastSpeech2Loss(preprocess_config, model_config)

        self.log_dir = log_dir
        self.result_dir = result_dir

        # All of the settings below are for few-shot validation
        adaptation_lr = self.algorithm_config["adapt"]["task"]["lr"]
        self.learner = MAML(self.model, lr=adaptation_lr)

        self.d_target = False
        self.reference_prosody = True

    def setup(self, stage=None):
        self.log_dir = os.path.join(self.trainer.log_dir, self.trainer.state.fn)
        self.result_dir = os.path.join(self.trainer.log_dir, self.trainer.state.fn)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def clone_learner(self, inference=False):
        learner = self.learner.clone()
        if inference:
            detach_module(learner.module, keep_requires_grad=True)
        learner.inner_freeze()
        for module in self.algorithm_config["adapt"]["modules"]:
            learner.inner_unfreeze(module)
        return learner

    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adapt_steps=5, learner=None, train=True):
        """ Inner adaptation or fine-tuning. """
        if learner is None:
            learner = self.clone_learner(inference=not train)
        learner.train()

        # Adapt the classifier
        sup_batch = batch[0][0][0]
        # ref_psd = sup_batch[9].unsqueeze(2) if self.reference_prosody else None
        ref_psd = (sup_batch["p_targets"].unsqueeze(2)
                   if self.reference_prosody else None)
        first_order = not train
        for _ in range(adapt_steps):
            # preds = learner(*sup_batch[2:], reference_prosody=ref_psd)
            preds = learner(**sup_batch, reference_prosody=ref_psd)
            train_error = self.loss_func(sup_batch, preds)
            learner.adapt_(
                train_error[0],
                first_order=first_order, allow_unused=False, allow_nograd=True
            )
        return learner

    def meta_learn(self, batch, adapt_steps=None, learner=None, train=True,
                   synth=False, recon=False):
        """ Inner-adapt/fine-tune + recon_predictions + (outer-/eval-)loss.

        Args:
            synth: Also return synthesized output w/o using prosody label.
            recon: Using prosody label to forward instead of predicted by VA.
        """
        learner = self.adapt(batch, adapt_steps, learner=learner, train=train)
        learner.eval()

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        # ref_psd = qry_batch[9].unsqueeze(2) if self.reference_prosody else None
        ref_psd = (qry_batch["p_targets"].unsqueeze(2)
                   if self.reference_prosody else None)

        # Evaluating the adapted model with ground_truth d,p,e (recon_preds)
        # Use speaker embedding of support set
        with torch.set_grad_enabled(train):
            predictions = learner(
                speaker_args=sup_batch["speaker_args"],
                texts=qry_batch["texts"],
                src_lens=qry_batch["src_lens"],
                max_src_len=qry_batch["max_src_len"],
                mels=qry_batch["mels"],
                mel_lens=qry_batch["mel_lens"],
                max_mel_len=qry_batch["max_mel_len"],
                p_targets=qry_batch["p_targets"],
                e_targets=qry_batch["e_targets"],
                d_targets=qry_batch["d_targets"],
                average_spk_emb=True,
                reference_prosody=ref_psd,
            )
            valid_error = self.loss_func(qry_batch, predictions)
            if not train:
                for element in predictions:
                    try:
                        element.detach_()
                    except:
                        pass

        if not synth:
            return learner, valid_error, predictions

        # Synthesize without ground-truth prosody during inference
        with torch.no_grad():
            synth_predictions = learner(
                speaker_args=sup_batch["speaker_args"],
                texts=qry_batch["texts"],
                src_lens=qry_batch["src_lens"],
                max_src_len=qry_batch["max_src_len"],
                mels=qry_batch["mels"] if self.d_target else None,
                mel_lens=qry_batch["mel_lens"] if self.d_target else None,
                max_mel_len=qry_batch["max_mel_len"] if self.d_target else None,
                d_targets=qry_batch["d_targets"] if self.d_target else None,
                average_spk_emb=True,
                reference_prosody=ref_psd
            )
            for element in synth_predictions:
                try:
                    element.detach_()
                except:
                    pass
        if recon:
            return learner, valid_error, predictions, synth_predictions

        return learner, valid_error, synth_predictions

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        LibriTTS subset n_spk & spk_id:
            - train-clean-100: 247 (spk_id 0 ~ 246)
            - train-clean-360: 904 (spk_id 247 ~ 1150)
            - train-other-500: 1160 (spk_id 1151 ~ 2310)
            - dev-clean: 40 (spk_id 2311 ~ 2350)
            - test-clean: 39 (spk_id 2351 ~ 2389)
        """
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        changes = {"skip": [], "drop": [], "replace": [], "miss": []}

        if 'model.speaker_emb.weight' in state_dict:
            # ==================================================================
            # Load from oldest version:
            #   model.speaker_emb.weight -> model.speaker_emb.model.weight
            # ------------------------------------------------------------------
            old_key = "model.speaker_emb.weight"
            new_key = "model.speaker_emb.model.weight"
            assert new_key in model_state_dict
            assert new_key not in state_dict
            state_dict[new_key] = state_dict.pop(old_key)
            changes["replace"].append([old_key, new_key])
            # ------------------------------------------------------------------

        for k in state_dict:
            if k not in model_state_dict:
                changes["drop"].append(k)
                continue

            # ==================================================================
            # k in model_state_dict
            # ------------------------------------------------------------------
            if state_dict[k].shape == model_state_dict[k].shape:
                continue
            # ------------------------------------------------------------------

            # ==================================================================
            # k in model_state_dict
            # state_dict[k].shape != model_state_dict[k].shape
            # ------------------------------------------------------------------
            if k == "model.speaker_emb.model.weight":
                if self.preprocess_config["dataset"] == "LibriTTS":
                    # LibriTTS: testing emb already in ckpt
                    # Shape mismatch: version problem
                    # (train-clean-100 vs train-all)
                    assert self.algorithm_config["adapt"]["speaker_emb"] == "table"
                    assert (state_dict[k].shape[0] == 326
                            and model_state_dict[k].shape[0] == 2390), \
                        f"state_dict: {state_dict[k].shape}, model: {model_state_dict[k].shape}"
                    model_state_dict[k][:247] = state_dict[k][:247]
                    model_state_dict[k][-79:] = state_dict[k][-79:]
                else:
                    # Corpus mismatch
                    assert state_dict[k].shape[0] in [326, 2390]
                    assert self.algorithm_config["adapt"]["speaker_emb"] == "table"
                    if self.algorithm_config["adapt"]["test"].get(
                            "avg_train_spk_emb", False
                    ):
                        model_state_dict[k][:] = state_dict[k][:247].mean(dim=0)
                        # if self.local_rank == 0:
                        #     print("Average training speaker emb as testing emb")
                        self.print("Average training speaker emb as testing emb")
                    else:
                        # Other testing corpus with different num of spks: re-initialize
                        pass
            changes["skip"].append([
                k, model_state_dict[k].shape, state_dict[k].shape
            ])
            state_dict[k] = model_state_dict[k]
            # ------------------------------------------------------------------

        for k in model_state_dict:
            if k not in state_dict:
                changes["miss"].append(k)

        if self.local_rank == 0:
            self.print()
            for replaced in changes["replace"]:
                before, after = replaced
                self.print(f"Replace: {before}\n\t-> {after}")
            for skipped in changes["skip"]:
                k, required_shape, loaded_shape = skipped
                self.print(f"Skip parameter: {k}, \n" \
                           + f"\trequired shape: {required_shape}, " \
                           + f"loaded shape: {loaded_shape}")
            for dropped in changes["drop"]:
                del state_dict[dropped]
            for dropped in set(['.'.join(k.split('.')[:2]) for k in changes["drop"]]):
                self.print(f"Dropping parameter: {dropped}")
            for missed in set(['.'.join(k.split('.')[:2]) for k in changes["miss"]]):
                self.print(f"Missing parameter: {missed}")
            self.print()

        if sum([len(v) for v in changes.values()]):
            # Changed
            checkpoint.pop("optimizer_states", None)

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

    def _print_mem_diff(self, st, prefix):
        if VERBOSE:
            self.print(f"{prefix}: {time.time() - st} sec")
            self.print(pd.DataFrame(self._mem_stats_diff()).T
                    .to_string(header=True, index=True))
            # self.print(torch.cuda.memory_summary(self.device, True))

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
