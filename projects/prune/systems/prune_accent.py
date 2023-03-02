#!/usr/bin/env python3

import os
import yaml
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from collections import defaultdict

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy

from src.utils.tools import load_yaml
from lightning.systems.base_adapt.base import BaseAdaptorSystem
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.utils import loss2dict
from projects.xvec.model import XvecTDNN


class PruneAccentSystem(BaseAdaptorSystem):
    """
    Inherit from BaseAdaptorSystem for `on_load_checkpoint`.

    This class mainly covers:
    - Support/Query set forward step
    - Validation forward step
    - Evaluate and save the following metrics:
        - speaker/accent accuracy
        - number of fine-tuning steps

    This class itself does not cover the pruning algorithm. To train with
    Lottery Ticket Hypothesis, use together with
    `pytorch_lightning.callbacks.pruning.ModelPruning`.
    """
    def __init__(self,
                 preprocess_config: Union[dict, str],
                 model_config: Union[dict, str],
                 algorithm_config: Union[dict, str],
                 ckpt_path: str,
                 qry_patience: int,
                 log_dir: str = None,
                 result_dir: str = None,
                 accent_ckpt: str = "output/xvec/lightning_logs/version_88/checkpoints/epoch=199-step=164200.ckpt",
                 speaker_ckpt: str= "output/xvec/lightning_logs/version_99/checkpoints/epoch=49-step=377950.ckpt"):
        """
        Args:
            preprocess_config: preprocess config
            model_config: model config
            algorithm_config: algorithm config
            ckpt_path: pretrained checkpoint path
            qry_patience: number of steps for early stopping by eval on query set
            log_dir:
            result_dir:
            accent_ckpt: checkpoint path of pretrained accent classifier
            speaker_ckpt: checkpoint path of pretrained speaker classifier

        Attributes:
            model: FastSpeech2 model (load from ckpt_path).
            loss_func: FastSpeech2 loss function.
            d_target (bool): Whether use ground-truth duration for inference. Default ``False``.
            reference_prosody (bool): Whether use ground-truch prosody for prosody encoder. Only effective when algorithm_config["adapt"]["AdaLN"]["prosody"] exists. Default: ``True``.
            override_kwargs: kwargs to override during inference.
            avg_train_spk_emb (bool): Initialize new speaker embeddings with the average of pretrained speaker embeddings. Default: ``True``.
            log_metrics_per_step (bool): self.log_dict(metrics) in every train/val step. Default: ``False``.
        """
        pl.LightningModule.__init__(self)

        if isinstance(preprocess_config, str):
            preprocess_config = load_yaml(preprocess_config)
        if isinstance(model_config, str):
            model_config = load_yaml(model_config)
        if isinstance(algorithm_config, str):
            algorithm_config = load_yaml(algorithm_config)

        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.algorithm_config = algorithm_config
        self.save_hyperparameters()

        self.model = FastSpeech2(preprocess_config, model_config, algorithm_config)
        self.loss_func = FastSpeech2Loss(preprocess_config, model_config)

        self.log_dir = log_dir
        self.result_dir = result_dir
        self.accent_ckpt = accent_ckpt
        self.speaker_ckpt = speaker_ckpt

        # All of the settings below are for few-shot validation
        # adaptation_lr = self.algorithm_config["adapt"]["task"]["lr"]
        # self.learner = MAML(self.model, lr=adaptation_lr)

        self.d_target = False
        self.reference_prosody = True
        self.override_kwargs = {k: None for k in {"p_targets", "e_targets"}}
        if not self.d_target:
            self.override_kwargs.update({
                k: None for k in {"mels", "mel_lens", "max_mel_len", "d_targets"}
            })

        self.qry_patience = qry_patience

        self.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)

        self.avg_train_spk_emb = True

        self.log_metrics_per_step = False

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1)
        self.opt_init_state = self.optimizer.state_dict

        return self.optimizer

    def on_fit_start(self,):
        if self.avg_train_spk_emb:
            with torch.no_grad():
                try:
                    self.model.speaker_emb.model.weight[2311:] = \
                        self.model.speaker_emb.model.weight[:2311].mean(dim=0)
                except:
                    self.model.speaker_emb.model.weight_orig[2311:] = \
                        self.model.speaker_emb.model.weight_orig[:2311].mean(dim=0)

        self.qry_stats = []

        # initialize eval_model.accent here to avoid checkpoint loading error
        self.eval_model = torch.nn.ModuleDict({
            "accent": XvecTDNN.load_from_checkpoint(self.accent_ckpt),
            "speaker": XvecTDNN.load_from_checkpoint(self.speaker_ckpt),
        }).to(self.device).requires_grad_(False).eval()

        self.eval_acc = torch.nn.ModuleDict({
            "accent": torch.nn.ModuleDict({
                "train_sup": Accuracy(ignore_index=-100),
                "train_qry": Accuracy(ignore_index=-100),
                "val_start": Accuracy(ignore_index=-100),
                "val_end": Accuracy(ignore_index=-100),
            }),
            "speaker": torch.nn.ModuleDict({
                "train_sup": Accuracy(ignore_index=-100),
                "train_qry": Accuracy(ignore_index=-100),
                "val_start": Accuracy(ignore_index=-100),
                "val_end": Accuracy(ignore_index=-100),
            }),
        }).to(self.device)

    def on_save_checkpoint(self, checkpoint):
        keys = sorted(checkpoint["state_dict"].keys())
        for key in keys:
            if key.startswith("eval_"):
                del checkpoint["state_dict"][key]

    def on_train_epoch_start(self) -> None:
        self.qry_stats.append({
            "epoch": self.trainer.current_epoch,
            "start_step": self.global_step,
        })

    def on_train_epoch_end(self,):
        # self.optimizer.load_state_dict(self.opt_init_state)
        self.optimizer.__setstate__({'state': defaultdict(dict)})

    def _check_epoch_done(self, batch_idx):
        if (batch_idx > self.qry_stats[-1].get("batch_idx", 0) + self.qry_patience
                and self.qry_stats[-1].get("accent_acc", 0) > 0.99
                and self.qry_stats[-1].get("accent_prob", 0) > 0.99
                and self.qry_stats[-1].get("speaker_acc", 0) > 0.95
                and self.qry_stats[-1].get("speaker_prob", 0) > 0.95):
            return True
        return False

    def on_train_batch_start(self, batch, batch_idx):
        """Log global_step to progress bar instead of GlobalProgressBar
        callback.
        """
        if self._check_epoch_done(batch_idx):
            return
        self.log("batch_idx", batch_idx, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.
        Including k-shot support set and q-shot query set forward + evaluate.
        """
        if self._check_epoch_done(batch_idx):
            return None
        batch = batch[0]
        sup_batch, qry_batch = batch["sup"], batch["qry"]

        # Query set forwarding & evaluating accents
        qry_batch["speaker_args"] = sup_batch["speaker_args"]
        with torch.no_grad():
            qry_loss, qry_output = self.subset_step(qry_batch, "train_qry", True)

        qry_results = {
            "total_loss": qry_loss[0].item(),
            **{k: v for k, v in qry_output.items()
               if k not in {"losses", "preds"}},
        }
        to_update = (
            qry_results["accent_acc"] >= self.qry_stats[-1]["accent_acc"]
            and qry_results["accent_prob"] > self.qry_stats[-1]["accent_prob"]
        ) or (
            qry_results["speaker_acc"] >= self.qry_stats[-1]["speaker_acc"]
            and qry_results["speaker_prob"] > self.qry_stats[-1]["speaker_prob"]
        )
        if "batch_idx" not in self.qry_stats[-1] or to_update:
            self.qry_stats[-1].update({"batch_idx": batch_idx, **qry_results})

        if self.log_metrics_per_step:
            prev_update = batch_idx - self.qry_stats[-1]["batch_idx"]
            self.log("prev_update", prev_update, prog_bar=True)
        self.print("qry", batch_idx, qry_results)

        # Support set forwarding & evaluating accents
        sup_loss, sup_output = self.subset_step(sup_batch, "train_sup", False)

        return {
            "loss": sup_loss[0],
            **{f"sup_{k}": v for k, v in sup_output.items()},
            **{f"qry_{k}": v for k, v in qry_output.items()},
        }

    @torch.no_grad()
    def eval_accent(self, targets, mels, eval_accent_acc):
        self.eval_model.accent.freeze()
        self.eval_model.accent.eval()
        mask = targets >= 0
        with torch.no_grad():
            _acc_logits = self.eval_model.accent(mels.transpose(1, 2))
            _acc_prob = F.softmax(_acc_logits, dim=-1)[
                torch.arange(_acc_logits.shape[0]).to(self.device)[mask],
                targets[mask]]
        accent_acc = eval_accent_acc(_acc_logits, targets)
        return _acc_logits, _acc_prob, accent_acc

    @torch.no_grad()
    def eval_speaker(self, targets, mels, eval_speaker_acc):
        self.eval_model.speaker.freeze()
        self.eval_model.speaker.eval()
        mask = targets >= 0
        with torch.no_grad():
            _spk_logits = self.eval_model.speaker(mels.transpose(1, 2))
            _spk_prob = F.softmax(_spk_logits, dim=-1)[
                torch.arange(_spk_logits.shape[0]).to(self.device)[mask],
                targets[mask]]
        speaker_acc = eval_speaker_acc(_spk_logits, targets)
        return _spk_logits, _spk_prob, speaker_acc

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        epoch_start = (self.trainer.state.stage != "sanity_check"
                       and "batch_idx" not in self.qry_stats[-1])
        tag = f"val_start" if epoch_start else "val_end"

        val_loss, val_output = self.subset_step(batch, tag, True)

        val_results = {
            "total_loss": val_loss[0].item(),
            "accent_acc": val_output["accent_acc"].item(),
            "accent_prob": val_output["accent_prob"].mean().item(),
            "speaker_acc": val_output["speaker_acc"].item(),
            "speaker_prob": val_output["speaker_prob"].mean().item(),
        }
        self.print("val", batch_idx, val_results)

        return {
            "epoch_start": epoch_start,
            **val_output,
        }

    def subset_step(self, batch, subset_name, synth=False, _model=None,
                    eval=True):
        r"""
        Args:
            _model: Use copied model instead of self.
        """
        model = _model if _model is not None else self
        preds = model(
            **batch,
            average_spk_emb=True,
            reference_prosody=(batch["p_targets"].unsqueeze(2)
                               if self.reference_prosody else None),
        )
        loss = self.loss_func(batch, preds)
        loss_dict = loss2dict(loss)
        if synth:
            # synth w/o ground-truth prosody
            preds = model(
                **{**batch, **self.override_kwargs},
                average_spk_emb=True,
                reference_prosody=(batch["p_targets"].unsqueeze(2)
                                if self.reference_prosody else None),
            )
        output = {
            "losses": [l.detach() for l in loss],
            "preds": [o.detach() for o in preds],
        }

        if eval:
            accent_logits, accent_prob, accent_acc = self.eval_accent(
                batch["accents"], preds[1], self.eval_acc.accent[subset_name])
            speaker_logits, speaker_prob, speaker_acc = self.eval_speaker(
                batch["speakers"], preds[1],
                self.eval_acc.speaker[subset_name])
            loss_dict.update({
                "accent_prob": accent_prob.mean(),
                "accent_acc": self.eval_acc.accent[subset_name],
                "speaker_prob": speaker_prob.mean(),
                "speaker_acc": self.eval_acc.speaker[subset_name],
            })

            # Support set logging
            if subset_name[:3] == "val":
                # sample-wise accent accuracy
                _accent_acc = torch.argmax(accent_logits, dim=-1) == batch["accents"]
                # sample-wise speaker accuracy
                _speaker_acc = torch.argmax(speaker_logits, dim=-1) == batch["speakers"]
                if self.log_metrics_per_step:
                    self.log_dict(
                        {f"{k}-val/epoch={self.current_epoch}": v
                        for k, v in loss_dict.items()},
                        sync_dist=True, batch_size=len(batch["ids"]))
                output.update({
                    # "losses": [l.detach() for l in loss],
                    # "preds": [o.detach() for o in preds],
                    "accent_prob": accent_prob,
                    "accent_acc": accent_acc,
                    "_accent_acc": _accent_acc,
                    "speaker_prob": speaker_prob,
                    "speaker_acc": speaker_acc,
                    "_speaker_acc": _speaker_acc,
                })
            else:
                # train_sup / train_qry
                if _model is None and self.log_metrics_per_step:
                    self.log_dict(
                        {f"{k}-{subset_name}/epoch={self.current_epoch}": v
                        for k, v in loss_dict.items()},
                        sync_dist=True, batch_size=len(batch["ids"]))
                output.update({
                    # "losses": [l.detach() for l in loss],
                    # "preds": [o.detach() for o in preds],
                    "accent_prob": accent_prob.mean().item(),
                    "accent_acc": accent_acc.item(),
                    "speaker_prob": speaker_prob.mean().item(),
                    "speaker_acc": speaker_acc.item(),
                })

        return loss, output

