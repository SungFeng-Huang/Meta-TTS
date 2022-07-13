#!/usr/bin/env python3

import os
import pandas as pd
from collections import defaultdict
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelCheckpoint
from torchmetrics import Accuracy

from .base import BaseAdaptorSystem
from lightning.model import FastSpeech2Loss, FastSpeech2
# from lightning.callbacks.saver import FitSaver
# from lightning.optimizer import get_optimizer
# from lightning.scheduler import get_scheduler
from lightning.utils import loss2dict
from lightning.model.xvec import XvecTDNN


class PruneAccentSystem(BaseAdaptorSystem):
    def __init__(self,
                 preprocess_config: dict,
                 model_config: dict,
                 algorithm_config: dict,
                 ckpt_path: str,
                 qry_patience: int,
                 log_dir: str = None,
                 result_dir: str = None):
        pl.LightningModule.__init__(self)
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.algorithm_config = algorithm_config
        self.save_hyperparameters()

        self.model = FastSpeech2(preprocess_config, model_config, algorithm_config)
        self.loss_func = FastSpeech2Loss(preprocess_config, model_config)

        self.log_dir = log_dir
        self.result_dir = result_dir

        # All of the settings below are for few-shot validation
        # adaptation_lr = self.algorithm_config["adapt"]["task"]["lr"]
        # self.learner = MAML(self.model, lr=adaptation_lr)

        self.d_target = False
        self.reference_prosody = True

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
                self.model.speaker_emb.model.weight[2311:] = \
                    self.model.speaker_emb.model.weight[:2311].mean(dim=0)

        self.qry_stats = []

        # initialize eval_accent_model here to avoid checkpoint loading error
        accent_xvec_ckpt_path = "output/xvec/lightning_logs/version_88/checkpoints/epoch=199-step=164200.ckpt"
        spk_xvec_ckpt_path = "output/xvec/lightning_logs/version_99/checkpoints/epoch=49-step=377950.ckpt"
        self.eval_model = torch.nn.ModuleDict({
            "accent": XvecTDNN.load_from_checkpoint(accent_xvec_ckpt_path),
            "speaker": XvecTDNN.load_from_checkpoint(spk_xvec_ckpt_path),
        }).to(self.device)
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
        self.eval_accent_model = self.eval_model.accent
        self.eval_accent_acc = self.eval_acc.accent
        self.eval_accent_model.freeze()
        self.eval_accent_model.eval()
        self.eval_speaker_model = self.eval_model.speaker
        self.eval_speaker_acc = self.eval_acc.speaker
        self.eval_speaker_model.freeze()
        self.eval_speaker_model.eval()

    def on_save_checkpoint(self, checkpoint):
        keys = sorted(checkpoint["state_dict"].keys())
        for key in keys:
            if key.startswith("eval_"):
                del checkpoint["state_dict"][key]

    def on_train_epoch_start(self,):
        self.qry_stats.append({
            "epoch": self.trainer.current_epoch,
            "start_step": self.global_step,
        })

    def on_train_epoch_end(self,):
        # self.optimizer.load_state_dict(self.opt_init_state)
        self.optimizer.__setstate__({'state': defaultdict(dict)})

    def _check_epoch_done(self, batch_idx):
        if (batch_idx > self.qry_stats[-1].get("batch_idx", 0) + self.qry_patience
                and self.qry_stats[-1].get("qry_accent_acc", 0) == 1
                and self.qry_stats[-1].get("qry_accent_prob", 0) > 0.99
                and self.qry_stats[-1].get("qry_speaker_acc", 0) == 1
                and self.qry_stats[-1].get("qry_speaker_prob", 0) > 0.99):
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
        qry_loss, qry_preds = self.qry_forward(batch)
        _, qry_accent_prob, qry_accent_acc = self.eval_accent(
            qry_batch["accents"], qry_preds[1], self.eval_accent_acc["train_qry"])
        _, qry_speaker_prob, qry_speaker_acc = self.eval_speaker(
            qry_batch["speakers"], qry_preds[1], self.eval_speaker_acc["train_qry"])

        qry_results = {
            "qry_accent_acc": qry_accent_acc.item(),
            "qry_accent_prob": qry_accent_prob.mean().item(),
            "qry_speaker_acc": qry_speaker_acc.item(),
            "qry_speaker_prob": qry_speaker_prob.mean().item(),
            "qry_total_loss": qry_loss[0].item(),
        }
        if ("batch_idx" not in self.qry_stats[-1]
                or (
                    qry_accent_acc >= self.qry_stats[-1]["qry_accent_acc"]
                    and qry_accent_prob.mean() > self.qry_stats[-1]["qry_accent_prob"]
                ) or (
                    qry_speaker_acc >= self.qry_stats[-1]["qry_speaker_acc"]
                    and qry_speaker_prob.mean() > self.qry_stats[-1]["qry_speaker_prob"]
                )):
            self.qry_stats[-1].update({"batch_idx": batch_idx, **qry_results})
        if self.log_metrics_per_step:
            self.log("prev_update",
                    batch_idx - self.qry_stats[-1]["batch_idx"],
                    prog_bar=True)
        self.print(batch_idx, qry_results)

        # Query set logging
        qry_loss_dict = loss2dict(qry_loss)
        qry_loss_dict.update({
            "accent_prob": qry_accent_prob.mean(),
            "accent_acc": self.eval_accent_acc["train_qry"],
            "speaker_prob": qry_speaker_prob.mean(),
            "speaker_acc": self.eval_speaker_acc["train_qry"],
        })
        if self.log_metrics_per_step:
            self.log_dict(
                {f"{k}-train_qry/epoch={self.current_epoch}": v for k, v in qry_loss_dict.items()},
                sync_dist=True, batch_size=len(qry_batch["ids"]))

        # Support set forwarding & evaluating accents
        sup_preds = self(
            **sup_batch,
            average_spk_emb=True,
            reference_prosody=(sup_batch["p_targets"].unsqueeze(2)
                               if self.reference_prosody else None),
        )
        sup_loss = self.loss_func(sup_batch, sup_preds)
        _, sup_accent_prob, sup_accent_acc = self.eval_accent(
            sup_batch["accents"], sup_preds[1], self.eval_accent_acc["train_sup"])
        _, sup_speaker_prob, sup_speaker_acc = self.eval_speaker(
            sup_batch["speakers"], sup_preds[1], self.eval_speaker_acc["train_sup"])

        # Support set logging
        sup_loss_dict = loss2dict(sup_loss)
        sup_loss_dict.update({
            "accent_prob": sup_accent_prob.mean(),
            "accent_acc": self.eval_accent_acc["train_sup"],
            "speaker_prob": sup_speaker_prob.mean(),
            "speaker_acc": self.eval_speaker_acc["train_sup"],
        })
        if self.log_metrics_per_step:
            self.log_dict(
                {f"{k}-train_sup/epoch={self.current_epoch}": v
                for k, v in sup_loss_dict.items()},
                sync_dist=True, batch_size=len(sup_batch["ids"]))

        return {
            "loss": sup_loss[0],
            "sup_losses": [l.detach() for l in sup_loss],
            "sup_preds": [o.detach() for o in sup_preds],
            "sup_accent_prob": sup_accent_prob.mean().item(),
            "sup_accent_acc": sup_accent_acc.item(),
            "sup_speaker_prob": sup_speaker_prob.mean().item(),
            "sup_speaker_acc": sup_speaker_acc.item(),
            "qry_losses": [l.detach() for l in qry_loss],
            "qry_preds": [o.detach() for o in qry_preds],
            "qry_accent_prob": qry_accent_prob.mean().item(),
            "qry_accent_acc": qry_accent_acc.item(),
            "qry_speaker_prob": qry_speaker_prob.mean().item(),
            "qry_speaker_acc": qry_speaker_acc.item(),
        }

    @torch.no_grad()
    def eval_accent(self, targets, mels, eval_accent_acc):
        self.eval_accent_model.freeze()
        self.eval_accent_model.eval()
        mask = targets >= 0
        with torch.no_grad():
            _acc_logits = self.eval_accent_model(mels.transpose(1, 2))
            _acc_prob = F.softmax(_acc_logits, dim=-1)[
                torch.arange(_acc_logits.shape[0]).to(self.device)[mask],
                targets[mask]]
        accent_acc = eval_accent_acc(_acc_logits, targets)
        return _acc_logits, _acc_prob, accent_acc

    @torch.no_grad()
    def eval_speaker(self, targets, mels, eval_speaker_acc):
        self.eval_speaker_model.freeze()
        self.eval_speaker_model.eval()
        mask = targets >= 0
        with torch.no_grad():
            _spk_logits = self.eval_speaker_model(mels.transpose(1, 2))
            _spk_prob = F.softmax(_spk_logits, dim=-1)[
                torch.arange(_spk_logits.shape[0]).to(self.device)[mask],
                targets[mask]]
        speaker_acc = eval_speaker_acc(_spk_logits, targets)
        return _spk_logits, _spk_prob, speaker_acc

    @torch.no_grad()
    def qry_forward(self, batch):
        sup_batch, qry_batch = batch["sup"], batch["qry"]
        # Forward with ground-truth prosody
        _qry_preds = self(
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
            reference_prosody=(qry_batch["p_targets"].unsqueeze(2)
                               if self.reference_prosody
                               else None),
        )
        qry_loss = self.loss_func(qry_batch, _qry_preds)
        # Forward w/o ground-truth prosody
        # if self.d_target is True, constrained with ground-truth duration
        qry_preds = self(
            speaker_args=sup_batch["speaker_args"],
            texts=qry_batch["texts"],
            src_lens=qry_batch["src_lens"],
            max_src_len=qry_batch["max_src_len"],
            mels=qry_batch["mels"] if self.d_target else None,
            mel_lens=qry_batch["mel_lens"] if self.d_target else None,
            max_mel_len=qry_batch["max_mel_len"] if self.d_target else None,
            p_targets=None,
            e_targets=None,
            d_targets=qry_batch["d_targets"] if self.d_target else None,
            average_spk_emb=True,
            reference_prosody=(qry_batch["p_targets"].unsqueeze(2)
                               if self.reference_prosody
                               else None),
        )
        for element in qry_preds:
            try:
                element.detach_()
            except:
                pass
        return qry_loss, qry_preds

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        epoch_start = (self.trainer.state.stage != "sanity_check"
                       and "batch_idx" not in self.qry_stats[-1])
        tag = f"val_start" if epoch_start else "val_end"

        _preds = self(
            **batch,
            average_spk_emb=True,
            reference_prosody=(batch["p_targets"].unsqueeze(2)
                               if self.reference_prosody
                               else None),
        )
        loss = self.loss_func(batch, _preds)
        preds = self(
            speaker_args=batch["speaker_args"],
            texts=batch["texts"],
            src_lens=batch["src_lens"],
            max_src_len=batch["max_src_len"],
            mels=batch["mels"] if self.d_target else None,
            mel_lens=batch["mel_lens"] if self.d_target else None,
            max_mel_len=batch["max_mel_len"] if self.d_target else None,
            p_targets=None,
            e_targets=None,
            d_targets=batch["d_targets"] if self.d_target else None,
            average_spk_emb=True,
            reference_prosody=(batch["p_targets"].unsqueeze(2)
                               if self.reference_prosody
                               else None),
        )
        accent_logits, accent_prob, accent_acc = self.eval_accent(
            batch["accents"], preds[1], self.eval_accent_acc[tag])
        # sample-wise accent accuracy
        _accent_acc = torch.argmax(accent_logits, dim=-1) == batch["accents"]
        speaker_logits, speaker_prob, speaker_acc = self.eval_speaker(
            batch["speakers"], preds[1], self.eval_speaker_acc[tag])
        # sample-wise speaker accuracy
        _speaker_acc = torch.argmax(speaker_logits, dim=-1) == batch["speakers"]

        # Query set logging
        loss_dict = loss2dict(loss)
        loss_dict.update({
            "accent_prob": accent_prob.mean(),
            "accent_acc": self.eval_accent_acc[tag],
            "speaker_prob": speaker_prob.mean(),
            "speaker_acc": self.eval_speaker_acc[tag],
        })
        self.log_dict(
            {f"{k}-val/{tag}": v for k, v in loss_dict.items()},
            sync_dist=True, batch_size=len(batch["ids"]))

        val_results = {
            "val_accent_acc": accent_acc.item(),
            "val_accent_prob": accent_prob.mean().item(),
            "val_speaker_acc": speaker_acc.item(),
            "val_speaker_prob": speaker_prob.mean().item(),
            "val_total_loss": loss[0].item(),
        }
        self.print(batch_idx, val_results)

        return {
            "epoch_start": epoch_start,
            # "losses": [l.detach() for l in loss],
            "preds": [o.detach() for o in preds],
            "accent_prob": accent_prob,
            "accent_acc": accent_acc,
            "_accent_acc": _accent_acc,
            "speaker_prob": speaker_prob,
            "speaker_acc": speaker_acc,
            "_speaker_acc": _speaker_acc,
        }
