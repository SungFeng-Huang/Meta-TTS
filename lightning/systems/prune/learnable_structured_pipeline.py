#!/usr/bin/env python3

import pandas as pd
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Any, Optional, Union, List
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

import torch
from torch.nn.utils.prune import L1Unstructured, RandomUnstructured, BasePruningMethod
from torchmetrics import Accuracy
from pytorch_lightning.callbacks.pruning import ModelPruning, _MODULE_CONTAINERS
from learn2learn.utils import clone_module, update_module, detach_module, clone_parameters

from lightning.scheduler import get_scheduler
from lightning.systems.prune.prune_accent import PruneAccentSystem
from lightning.utils import loss2dict
from .utils import global_learnable_unstructured, LogitsAndMasks, setup_structured_prune
from transformer.Layers import FFTBlock, MultiHeadAttention, PositionwiseFeedForward
from projects.xvec.model import XvecTDNN


torch.autograd.set_detect_anomaly(True)

class LearnableStructuredPruneSystem(PruneAccentSystem):
    """
    Inherit from BaseAdaptorSystem for `on_load_checkpoint` and `subset_step`.
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
                 speaker_ckpt: str= "output/xvec/lightning_logs/version_99/checkpoints/epoch=49-step=377950.ckpt",
                 target_sparsity: float = 0.,
                 mask_steps_per_epoch: int = 200,
                 verbose: int = 2,
                 lam_mask: dict = None,
                 mask_mode: str = "hard",
                 pipeline: list = None,
                 spk_emb_subset_weight_avg: bool = True,
                 libri_mask: bool = False,
                 ):
        super().__init__(
            preprocess_config=preprocess_config,
            model_config=model_config,
            algorithm_config=algorithm_config,
            ckpt_path=ckpt_path,
            qry_patience=qry_patience,
            log_dir=log_dir,
            result_dir=result_dir,
            accent_ckpt=accent_ckpt,
            speaker_ckpt=speaker_ckpt,
        )

        self.target_sparsity = target_sparsity
        # init_sparsity = 0.5 if target_sparsity == 0 else target_sparsity
        init_sparsity = 0.01
        self.logits_and_masks = LogitsAndMasks(
            self.model, mode=mask_mode, init_prob=1-init_sparsity, **lam_mask)
        setup_structured_prune(self.model)  # param -> param_orig

        # self.model.requires_grad_(False)

        self.automatic_optimization = False

        self.mask_steps_per_epoch = mask_steps_per_epoch
        self.verbose = verbose

        # Speaker embedding weighted average
        # Case 1: average all training speaker embedding
        #   Done in `on_fit_start()`
        # Case 2: average from a subset of training speakers
        #   Set self.spk_emb_subset_avg = True
        self.spk_emb_subset_weight_avg = spk_emb_subset_weight_avg
        if self.spk_emb_subset_weight_avg:
            # weight avg from subset of trained speaker
            n_spk = self.model.speaker_emb.model.weight_orig.shape[0]
            spk_avg_weight = torch.ones(n_spk)
            spk_avg_weight[2311:] = -float('inf')
            self.spk_emb_w_avg_weight_orig = torch.nn.Parameter(
                spk_avg_weight, requires_grad=False
            )
            spk_avg_probs = torch.ones(n_spk)
            spk_avg_probs[2311:] = 0
            self.spk_emb_w_avg_subset_logits = torch.nn.Parameter(
                torch.distributions.utils.probs_to_logits(
                    spk_avg_probs, is_binary=True
                ),
                requires_grad=True
            )

        self.inner_sched = True
        self.mask_mode = mask_mode
        self.pipeline = pipeline
        self.p_idx = 0
        self.libri_mask = libri_mask

        # self.logits_and_masks.sample_masks(training=False)    # for sanity_check or resume_ckpt
        # self.logits_and_masks.mask_model(self.model, detach_mask=True)
        # if self.spk_emb_subset_weight_avg:
        #     self._sample_spk_emb_w_avg_subset_mask(training=False)

    @property
    def pipeline_stage(self):
        return self.pipeline[self.p_idx]

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        mask_params = list(self.logits_and_masks.parameters())
        ft_params = list(self.model.parameters())
        if self.spk_emb_subset_weight_avg:
            mask_params = mask_params + [self.spk_emb_w_avg_subset_logits]
            ft_params = ft_params + [self.spk_emb_w_avg_weight_orig]
        self.mask_optimizer = torch.optim.Adam(mask_params, lr=1e2)
        self.mask_scheduler = get_scheduler(
            self.mask_optimizer,
            {
                "optimizer": {
                    "anneal_rate": 0.3,
                    "anneal_steps": [1000, 2000, 3000],
                    "warm_up_step": 600,
                },
            },
        )
        self.ft_optimizer = torch.optim.Adam(
            ft_params, lr=1e-1 if self.inner_sched else 3e-5)
        if self.inner_sched:
            self.ft_scheduler = get_scheduler(
                self.ft_optimizer,
                {
                    "optimizer": {
                        "anneal_rate": 0.3,
                        "anneal_steps": [500, 700],
                        "warm_up_step": 200,
                    },
                },
            )
            self.ft_sched_state_dict = self.ft_scheduler.state_dict()

            return (
                [self.mask_optimizer, self.ft_optimizer],
                [self.mask_scheduler, self.ft_scheduler]
            )
        return (
            [self.mask_optimizer, self.ft_optimizer],
            [self.mask_scheduler]
        )

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
            acc: torch.nn.ModuleDict({
                k: Accuracy(ignore_index=-100)
                for k in {
                    "train_mask", "train_sup", "train_qry",
                    "val",
                }
            }) for acc in {"accent", "speaker"}
        }).to(self.device)
        self.val_stats = {}

        self.trainer.datamodule.pipeline_stage = self.pipeline_stage

    def on_train_epoch_start(self) -> None:
        # EMA
        self.qry_stats.append({
            "epoch": self.trainer.current_epoch,
            "start_step": self.global_step,
        })
        # epoch stats
        self.qry_epoch_stats = []

        self.trainer.progress_bar_callback.main_progress_bar.set_description(
            f"Epoch {self.current_epoch} "
            f"[Stage {self.p_idx}/{len(self.pipeline)} {self.pipeline_stage}]")

    def on_train_batch_start(self, batch, batch_idx):
        """Log global_step to progress bar instead of GlobalProgressBar
        callback.
        """
        pass

    # def _setup_mask_then_ft_stage_change(self,):
    #     if self.spk_emb_subset_weight_avg:
    #         with torch.no_grad():
    #             self._sample_spk_emb_w_avg_subset_mask(training=False)
    #             self.spk_emb_w_avg_subset_mask.detach_()
    #
    # def _setup_ft_then_mask_stage_change(self,):
    #     if self.spk_emb_subset_weight_avg:
    #         with torch.no_grad():
    #             self._sample_spk_emb_w_avg_subset_mask(training=True)
    #             # self.spk_emb_w_avg_subset_mask.detach_()
    #     #TODO: check pipeline correctness

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.
        Including k-shot support set and q-shot query set forward + evaluate.
        """
        mask_opt, ft_opt = self.optimizers()
        if self.inner_sched:
            mask_sched, ft_sched = self.lr_schedulers()
        elif not self.inner_sched:
            mask_sched, ft_sched = self.lr_schedulers(), None
        if isinstance(batch, list):
            batch = batch[0]
            if "mask" in batch:
                mask_batch, qry_batch = batch["mask"], batch["qry"]
            if "sup" in batch:
                sup_batch, qry_batch = batch["sup"], batch["qry"]
        elif isinstance(batch, dict):
            assert self.libri_mask
            if "mask" in batch:
                mask_batch = batch["mask"]
            qry_batch = batch["ft"][0]["qry"]
            if "sup" in batch["ft"][0]:
                sup_batch = batch["ft"][0]["sup"]
        output = {}

        # if batch_idx < self.mask_steps_per_epoch:
        if self.pipeline_stage == "mask" or self.pipeline_stage == "joint":
            # qry_batch["speaker_args"] = mask_batch["speaker_args"]
            if not hasattr(self, "speaker_args"):
                self.register_buffer(
                    "speaker_args", torch.unique(qry_batch["speaker_args"]))

            loss, mask_output = self._train_mask_step(
                mask_batch, qry_batch, batch_idx, mask_opt, mask_sched)

            # Eval qry before epoch end
            # if self.trainer.is_last_batch:
            qry_loss, qry_output = self._qry_step(qry_batch, batch_idx, "m")
            if self.verbose in {1, 2}:
                if self.trainer.is_last_batch:
                    self.print(pd.DataFrame(self.qry_stats).set_index(["epoch"]))

            # Backward, update model
            mask_opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(mask_opt, gradient_clip_val=1.0)
            mask_opt.step()
            if mask_sched is not None:
                mask_sched.step()

            output.update({
                "mask": mask_output,
                "qry": qry_output,  # might be overwritten in joint
            })

        if self.pipeline_stage == "ft" or (
            self.pipeline_stage == "joint" and (batch_idx+1) % 10 == 0
        ):
            # Since more unstable, use EMA as ES criteria and check ES every step
            if self.pipeline_stage == "ft" and self._check_epoch_done(batch_idx):
                return None

            qry_batch["speaker_args"] = sup_batch["speaker_args"]
            if not hasattr(self, "speaker_args"):
                self.register_buffer(
                    "speaker_args", torch.unique(sup_batch["speaker_args"]))

            sup_loss, sup_output = self._train_ft_step(
                sup_batch, qry_batch, batch_idx, ft_opt, ft_sched)

            # Eval qry before check_epoch_done
            qry_loss, qry_output = self._qry_step(qry_batch, batch_idx, "ft")
            if self.verbose in {1, 2}:
                if self.trainer.is_last_batch or (
                    self.pipeline_stage == "ft"
                    and self._check_epoch_done(batch_idx+1)):
                    self.print(pd.DataFrame(self.qry_stats).set_index(["epoch"]))

            # Backward, update model
            if not self._check_epoch_done(batch_idx+1):
                ft_opt.zero_grad()
                self.manual_backward(sup_loss[0])
                self.clip_gradients(ft_opt, gradient_clip_val=1.0)
                ft_opt.step()
                if self.inner_sched and ft_sched is not None:
                    ft_sched.step()

            output.update({
                "sup": sup_output,
                "qry": qry_output,
            })

        return output

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        metrics = self.trainer.progress_bar_callback.get_metrics(
            self.trainer, self)
        if self.pipeline_stage == "mask":
            for k in list(metrics.keys()):
                if k.startswith("ft_"):
                    metrics.pop(k)
        elif self.pipeline_stage == "ft":
            for k in list(metrics.keys()):
                if k.startswith("m_"):
                    metrics.pop(k)
        self.trainer.progress_bar_callback.main_progress_bar.set_postfix(metrics)

    def on_validation_start(self) -> None:
        self.logits_and_masks.sample_masks(training=False)    # for sanity_check or ft
        self.logits_and_masks.mask_model(self.model, detach_mask=True)

        if self.spk_emb_subset_weight_avg:
            self._sample_spk_emb_w_avg_subset_mask(training=False)
            self._sample_weighted_speaker_emb()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Sanity: ??
        # if self.trainer.state.stage == "sanity_check":
        #     return

        output = {}
        df = []

        assert not self.model.training
        val_loss, val_output = self.subset_step(batch, "val", True)
        output["val"] = val_output

        _, _, density = self._get_mask_stats()
        sparsity = 1 - density
        output["val"]["sparsity"] = sparsity.item()

        val_results = {
            "total_loss": val_loss[0].item(),
            "accent_acc": val_output["accent_acc"].item(),
            "accent_prob": val_output["accent_prob"].mean().item(),
            "speaker_acc": val_output["speaker_acc"].item(),
            "speaker_prob": val_output["speaker_prob"].mean().item(),
        }
        df.append(
            {"stage": "val", "batch_idx": batch_idx, **val_results})
        self.log("val_l", val_results["total_loss"])
        self.log("val_acc_spk", val_results["speaker_acc"])
        self.log("val_acc_acnt", val_results["accent_acc"])

        if self.verbose in {2} and batch_idx == 0:
            self._print_model_stats(substage="")

        return output

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        # self._log_probs_histogram()
        if not self.log_metrics_per_step:
            accent_acc = self.eval_acc.accent.val.compute()
            self.eval_acc.accent.val.reset()
            speaker_acc = self.eval_acc.speaker.val.compute()
            self.eval_acc.speaker.val.reset()

            _, _, density = self._get_mask_stats()
            sparsity = 1 - density

            pre_df = {
                "epoch": self.current_epoch,
                "sparsity": sparsity.item(),
                "accent_acc": accent_acc.item(),
                "speaker_acc": speaker_acc.item(),
            }
            if self.trainer.state.stage not in self.val_stats:
                self.val_stats[self.trainer.state.stage] = []
            self.val_stats[self.trainer.state.stage].append(pre_df)
            if self.verbose in {1, 2}:
                self.print(
                    pd.concat(
                        {
                            stage: pd.DataFrame(
                                self.val_stats[stage]
                            ).set_index("epoch")
                            for stage in self.val_stats
                        },
                        names=["stage"]
                    )
                )

    def on_train_epoch_end(self,):
        # self.ft_optimizer.__setstate__({'state': defaultdict(dict)})
        # if self.inner_sched:
        #     # self.ft_scheduler.__setstate__({'state': defaultdict(dict)})
        #     self.ft_scheduler.load_state_dict(self.ft_sched_state_dict)
        # self.log("B_id", self.qry_stats[-1]["batch_idx"])
        metrics = self.trainer.progress_bar_callback.get_metrics(
            self.trainer, self)
        if self.pipeline_stage == "mask":
            for k in list(metrics.keys()):
                if k.startswith("ft_"):
                    metrics.pop(k)
        elif self.pipeline_stage == "ft":
            for k in list(metrics.keys()):
                if k.startswith("m_"):
                    metrics.pop(k)
        self.trainer.progress_bar_callback.main_progress_bar.set_postfix(metrics)

        if self.pipeline_stage == "mask":
            if self._evaluate_mask_stopping_criteria():
                self.early_stop_pipeline_stage = True
        elif self.pipeline_stage == "ft":
            if self._evaluate_ft_stopping_criteria():
                self.early_stop_pipeline_stage = True
        elif self.pipeline_stage == "joint":
            if (self._evaluate_mask_stopping_criteria()
                    and self._evaluate_ft_stopping_criteria()):
                self.early_stop_pipeline_stage = True

        if getattr(self, "early_stop_pipeline_stage", False):
            if self.p_idx == len(self.pipeline) - 1:
                self.trainer.should_stop = True
            else:
                prev_stage = self.pipeline_stage
                self.p_idx += 1
                curr_stage = self.pipeline_stage
                # if (prev_stage, curr_stage) == ("mask", "ft"):
                #     self._setup_mask_then_ft_stage_change()
                # elif (prev_stage, curr_stage) == ("ft", "mask"):
                #     self._setup_ft_then_mask_stage_change()
                self.early_stop_pipeline_stage = False
        self.trainer.datamodule.pipeline_stage = self.pipeline_stage

    def _train_mask_step(self, mask_batch, qry_batch, batch_idx, mask_opt, mask_sched=None):
        self.logits_and_masks.sample_masks(training=True)
        self.logits_and_masks.mask_model(self.model, detach_mask=False)

        if self.spk_emb_subset_weight_avg:
            self._sample_spk_emb_w_avg_subset_mask(training=True)
            self._sample_weighted_speaker_emb()

        # Forward
        mask_loss, mask_output = self.subset_step(
            mask_batch, "train_mask", False,
            eval=not getattr(self, "libri_mask", False))

        # Get stats
        n_params, n_zeros, density = self._get_mask_stats()
        sparsity = 1 - density
        entropy, entropy_prob = self._get_mask_entropy_stats()

        # Print/log stats
        if self.verbose in {2}:
            if batch_idx == 0:
                self._print_model_stats(substage="train_mask")
            self._print_mask_stats(
                batch_idx, n_zeros, n_params, sparsity,
                entropy, entropy_prob)

        # Update stats
        self._update_mask_stats(sparsity, entropy, entropy_prob)
        self.log("m_l", mask_loss[0], sync_dist=True, prog_bar=True)
        self.log("m_spar", sparsity, sync_dist=True, prog_bar=True)
        self.log("m_en", entropy, sync_dist=True)
        self.log("m_en_p", entropy_prob, sync_dist=True)
        if not getattr(self, "libri_mask", False):
            self.log("m_acc_acnt", mask_output["accent_acc"], sync_dist=True)
            self.log("m_acc_spk", mask_output["speaker_acc"], sync_dist=True)

        loss = mask_loss[0]
        if self.target_sparsity == 0:
            loss += density
        else:
            loss += max(density + self.target_sparsity, 1)

        return loss, mask_output

    def _train_ft_step(self, sup_batch, qry_batch, batch_idx, ft_opt,
                       ft_sched=None):
        self.logits_and_masks.mask_model(self.model, detach_mask=True)

        if self.spk_emb_subset_weight_avg:
            self._sample_spk_emb_w_avg_subset_mask(training=False)
            self._sample_weighted_speaker_emb()

        # Forward (sup, qry)
        sup_loss, sup_output = self.subset_step(sup_batch, "train_sup", False)

        # Print/log stats
        self.log("ft_sup_l",
                 sup_loss[0], sync_dist=True, prog_bar=True)
        self.log("ft_sup_acc_acnt",
                 sup_output["accent_acc"], sync_dist=True)
        self.log("ft_sup_acc_spk",
                 sup_output["speaker_acc"], sync_dist=True)

        return sup_loss, sup_output

    @torch.no_grad()
    def _qry_step(self, qry_batch, batch_idx, tag="ft"):
        qry_loss, qry_output = self.subset_step(
            qry_batch, "train_qry", False)

        # Collect query stats
        qry_results = {
            "batch_idx": batch_idx,
            "total_loss": qry_loss[0].item(),
            **{
                k: v for k, v in qry_output.items()
                if k not in {"losses", "preds"}
            },
        }

        # Update stats
        self.qry_epoch_stats.append(qry_results)
        if "batch_idx" not in self.qry_stats[-1]:
            self.qry_stats[-1].update(qry_results)
        else:
            for k in qry_results:
                if k in {"batch_idx"}:
                    self.qry_stats[-1]["batch_idx"] = batch_idx
                    continue
                moving_weight = 0.5
                self.qry_stats[-1][k] *= 1-moving_weight
                self.qry_stats[-1][k] += moving_weight * qry_results[k]
        if self.log_metrics_per_step:
            prev_update = batch_idx - self.qry_stats[-1]["batch_idx"]
            self.log("prev_update", prev_update, prog_bar=True)

        # Print/log stats
        self.log(f"{tag}_qry_l",
                 qry_loss[0], sync_dist=True, prog_bar=True)
        self.log(f"{tag}_qry_acc_acnt",
                 qry_output["accent_acc"], sync_dist=True)
        self.log(f"{tag}_qry_acc_spk",
                 qry_output["speaker_acc"], sync_dist=True)

        if self.verbose in {2}:
            if batch_idx == 0:
                self._print_model_stats(substage="train")
            qry_epoch_df = pd.DataFrame(self.qry_epoch_stats)
            qry_epoch_df = qry_epoch_df.set_index(["batch_idx"])
            qry_epoch_df = qry_epoch_df.to_string().splitlines()
            if batch_idx == 0:
                self.print(qry_epoch_df)
            else:
                self.print(qry_epoch_df[-1])

        return qry_loss, qry_output

    def _get_mask_stats(self):
        params = []
        for n, p in self.model.named_parameters():
            if n in {"variance_adaptor.pitch_bins",
                     "variance_adaptor.energy_bins"}:
                params.append(torch.ones_like(p))
                continue
            elif n[-5:] == "_orig":
                module_name, tensor_name = n[:-5].rsplit('.', 1)
                module = self.model.get_submodule(module_name)
                param = getattr(module, tensor_name)
                if self.mask_mode == "hard":
                    if n[:-5] == "speaker_emb.model.weight":
                        param = param[0]    # only 1 spk
                    params.append(param != 0)
                else:
                    mask = getattr(module, tensor_name+"_mask")
                    if n[:-5] == "speaker_emb.model.weight":
                        mask = mask[0]    # only 1 spk
                    params.append(mask.contiguous())
        masks = torch.nn.utils.parameters_to_vector(params)
        n_params = masks.numel()
        n_ones = masks.sum()
        n_zeros = n_params - n_ones
        density = n_ones / n_params
        return n_params, n_zeros, density

    def _log_probs_histogram(self):
        named_probs = []
        for n, p in self.logits_and_masks.named_parameters():
            if n[-7:] == "_logits":
                prob = torch.distributions.utils.logits_to_probs(
                    p, is_binary=True
                )
                named_probs.append((n, prob))
                self.logger.experiment.add_histogram(
                    "prob/"+n[:-7], prob*100, self.current_epoch)
        if self.spk_emb_subset_weight_avg:
            prob = torch.distributions.utils.logits_to_probs(
                self.spk_emb_w_avg_subset_logits, is_binary=True
            )
            named_probs.append(("spk_emb_w_avg_subset_logits", prob))
            self.logger.experiment.add_histogram(
                "prob/spk_emb_w_avg_subset", prob*100, self.current_epoch)
        probs = torch.nn.utils.parameters_to_vector([p for _, p in named_probs])
        self.logger.experiment.add_histogram("probs", probs*100, self.current_epoch)

    def _get_mask_entropy_stats(self):
        logits = torch.nn.utils.parameters_to_vector(
            [
                p
                for n, p in self.logits_and_masks.named_parameters()
                if n[-7:] == "_logits"
            ]
        )
        probs = torch.distributions.utils.logits_to_probs(
            logits, is_binary=True
        )
        entropy = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, probs
        )
        hard_probs = torch.ones_like(logits).masked_fill_(logits < 0, 0)
        hard_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, hard_probs
        )
        entropy_prob = torch.exp(-hard_entropy)
        return entropy, entropy_prob

    def _print_model_stats(self, substage):
        self.print(self.trainer.state.stage, substage)
        self.print(self.model.mel_linear.weight)
        self.print((self.model.mel_linear.weight == 0).sum().item())

    def _print_mask_stats(self, batch_idx, n_zeros, n_params, sparsity,
                          entropy, entropy_prob):
        self.print(
            batch_idx,
            f"Pruned: {n_zeros}/{n_params}",
            f"({sparsity:.2%})",
            f"Entropy: {entropy.item():.4f}",
            f"Ent_prob: {entropy_prob.item():.4f}")

    def _update_mask_stats(self, sparsity, entropy, entropy_prob):
        self.qry_stats[-1]["sparsity"] = sparsity.item()*100
        self.qry_stats[-1]["entropy"] = entropy.item()*100
        self.qry_stats[-1]["entropy_prob"] = entropy_prob.item()*100

    def _sample_spk_emb_w_avg_subset_mask(self, training=False):
        with torch.no_grad():
            spk_avg_mask = torch.ones_like(self.spk_emb_w_avg_subset_logits)
            spk_avg_mask[2311:] = 0
        if training:
            self.spk_emb_w_avg_subset_mask = torch.distributions.RelaxedBernoulli(
                torch.tensor([1.0]).to(self.device),
                logits=self.spk_emb_w_avg_subset_logits
            ).rsample() * spk_avg_mask
        else:
            self.spk_emb_w_avg_subset_mask = torch.ones_like(
                self.spk_emb_w_avg_subset_logits
            ).masked_fill_(
                self.spk_emb_w_avg_subset_logits < 0, 0
            ) * spk_avg_mask

    def _sample_weighted_speaker_emb(self):
        assert hasattr(self.model.speaker_emb.model, "weight_orig")
        assert hasattr(self.model.speaker_emb.model, "weight_mask")
        orig = self.model.speaker_emb.model.weight_orig
        mask = self.model.speaker_emb.model.weight_mask
        curr_spk_emb = orig * mask
        n_spk = curr_spk_emb.shape[0]

        spk_emb_w_avg_weight = torch.nn.functional.softmax(
            self.spk_emb_w_avg_weight_orig, dim=0
        ) * self.spk_emb_w_avg_subset_mask
        spk_emb_w_avg_weight = spk_emb_w_avg_weight / spk_emb_w_avg_weight.sum()
        new_spk_emb = spk_emb_w_avg_weight.view(1,-1) @ curr_spk_emb
        new_spk_emb = new_spk_emb.reshape(1, -1).expand(n_spk, -1)
        setattr(self.model.speaker_emb.model, "weight", new_spk_emb)

    def _evaluate_mask_stopping_criteria(self):
        #TODO
        early_stop_epoch = 1
        df = pd.DataFrame(self.qry_stats).set_index(["epoch"])
        df_ = df.head(-1*early_stop_epoch)
        _df = df.tail(early_stop_epoch)

        self.max_sparsity = df["sparsity"].max()
        recent_max_sparsities = _df["sparsity"].max()
        return (
            self.current_epoch > 0
            and recent_max_sparsities <= df_["sparsity"].max() + 0.1 * early_stop_epoch
        )
        # return self.current_epoch > 2
        # recent_min_sparsities = df[-5:]["sparsity"].min()
        # return self.current_epoch > 4 and recent_max_sparsities < recent_min_sparsities + 5

    def _evaluate_ft_stopping_criteria(self):
        #TODO
        early_stop_epoch = 3
        df = pd.DataFrame(self.qry_stats).set_index(["epoch"])
        df_ = df.head(-1*early_stop_epoch)
        _df = df.tail(early_stop_epoch)
        print(_df["speaker_acc"].max(), df_["speaker_acc"].max())
        print(_df["speaker_prob"].max(), df_["speaker_prob"].max())
        if (
            ("sparsity" in df and df["sparsity"].isna().sum() > 1 or self.current_epoch > 1)
            and _df["accent_acc"].max() <= df_["accent_acc"].max()
            and _df["accent_prob"].max() <= df_["accent_prob"].max()
            and _df["speaker_acc"].max() <= df_["speaker_acc"].max()
            and _df["speaker_prob"].max() <= df_["speaker_prob"].max()
        ) or (
            df[-1:]["accent_acc"].item() > 0.99
            and df[-1:]["accent_prob"].item() > 0.99
            and df[-1:]["speaker_acc"].item() > 0.9
            and df[-1:]["speaker_prob"].item() > 0.9
        ):
            return True
        return False

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        mask_state_dict = {}
        orig_state_dict = {}
        lam_state_dict = {"mask": {}, "logits": {}, "others": {}}
        model_state_dict = {"mask": {}, "orig": {}, "others": {}}

        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith("eval_model."):
                del checkpoint["state_dict"][key]

            elif key.startswith("logits_and_masks."):
                if key.endswith("_mask"):
                    lam_state_dict["mask"][key] = checkpoint["state_dict"].pop(key)
                elif key.endswith("_logits"):
                    lam_state_dict["logits"][key] = checkpoint["state_dict"].pop(key)
                else:
                    lam_state_dict["others"][key] = checkpoint["state_dict"].pop(key)

            elif key.startswith("model."):
                if key.endswith("_mask"):
                    model_state_dict["mask"][key] = checkpoint["state_dict"].pop(key)
                elif key.endswith("_orig"):
                    model_state_dict["orig"][key] = checkpoint["state_dict"].pop(key)
                else:
                    model_state_dict["others"][key] = checkpoint["state_dict"].pop(key)

        save_weights_only = self.trainer.checkpoint_callback.save_weights_only
        if not save_weights_only:
            checkpoint["lam_state_dict"] = lam_state_dict
        checkpoint["model_state_dict"] = model_state_dict

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.register_buffer("speaker_args", checkpoint["state_dict"]["speaker_args"])

        checkpoint["state_dict"].update({
            # **checkpoint["lam_state_dict"]["mask"],
            **checkpoint["lam_state_dict"]["logits"],
            **checkpoint["lam_state_dict"]["others"],
            # **checkpoint["model_state_dict"]["mask"],
            **checkpoint["model_state_dict"]["orig"],
            **checkpoint["model_state_dict"]["others"],
        })
        del checkpoint["lam_state_dict"]
        del checkpoint["model_state_dict"]
