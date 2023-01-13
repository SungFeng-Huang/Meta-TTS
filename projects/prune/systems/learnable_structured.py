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
from lightning.utils import loss2dict
from .prune_accent import PruneAccentSystem
from .utils import global_learnable_unstructured, LogitsAndMasks, structured_prune
from transformer.Layers import FFTBlock, MultiHeadAttention, PositionwiseFeedForward


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
                 validate: bool = False,
                 val_with_ft: bool = False,
                 mask_steps_per_epoch: int = 200,
                 verbose: int = 2,
                 lam_mask: dict = None,
                 mask_mode: str = "hard",
                 pipeline: list = None,
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

        self.copied = deepcopy(self.model)

        self.target_sparsity = target_sparsity
        # init_sparsity = 0.5 if target_sparsity == 0 else target_sparsity
        init_sparsity = 0.01
        self.logits_and_masks = LogitsAndMasks(
            self.model, mode=mask_mode, init_prob=1-init_sparsity, **lam_mask)
        structured_prune(self.model, self.logits_and_masks)
        structured_prune(self.copied, self.logits_and_masks)

        self.model.requires_grad_(False)

        # turn to false after debug
        self.copied.load_state_dict(self.model.state_dict(), strict=True)

        self.automatic_optimization = False

        self.validate = validate
        self.val_with_ft = val_with_ft
        self.mask_steps_per_epoch = mask_steps_per_epoch
        self.verbose = verbose

        self.spk_emb_weight_avg = True
        if self.spk_emb_weight_avg:
            n_spk = self.model.speaker_emb.model.weight_orig.shape[0]
            spk_avg_weight = torch.ones(n_spk)
            spk_avg_weight[2311:] = -float('inf')
            self.spk_emb_avg_weight_orig = torch.nn.Parameter(
                spk_avg_weight, requires_grad=False
            )
            self.spk_emb_avg_weight_copied = torch.nn.Parameter(
                spk_avg_weight, requires_grad=True
            )
            spk_avg_probs = torch.ones(n_spk)
            spk_avg_probs[2311:] = 0
            self.spk_emb_avg_weight_logits = torch.nn.Parameter(
                torch.distributions.utils.probs_to_logits(
                    spk_avg_probs, is_binary=True
                ),
                requires_grad=True
            )

        self.inner_sched = True
        self.mask_mode = mask_mode
        self.pipeline = pipeline
        self.pipeline_stage = pipeline[0]

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        mask_params = list(self.logits_and_masks.parameters())
        ft_params = list(self.copied.parameters())
        if self.spk_emb_weight_avg:
            mask_params = mask_params + [self.spk_emb_avg_weight_logits]
            ft_params = ft_params + [self.spk_emb_avg_weight_copied]
        self.mask_optimizer = torch.optim.Adam(mask_params, lr=1)
        self.mask_scheduler = get_scheduler(
            self.mask_optimizer,
            {
                "optimizer": {
                    "anneal_rate": 0.3,
                    "anneal_steps": [6000, 8000, 10000],
                    "warm_up_step": 1000,
                },
            },
        )
        self.ft_optimizer = torch.optim.Adam(
            ft_params, lr=1e-3 if self.inner_sched else 3e-5)
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
        super().on_fit_start()
        self.eval_accent_acc.update({
            "train_mask": Accuracy(ignore_index=-100).to(self.device),
        })
        self.eval_speaker_acc.update({
            "train_mask": Accuracy(ignore_index=-100).to(self.device),
        })
        self.val_stats = {}

        self.logits_and_masks.setup()
        self.logits_and_masks.setup_model(self.model)
        self.logits_and_masks.setup_model(self.copied)
        self._sample_spk_emb_avg_w_mask()
        self.trainer.dataloader.pipeline_stage = self.pipeline_stage

    def on_train_epoch_start(self) -> None:
        # EMA
        self.qry_stats.append({
            "epoch": self.trainer.current_epoch,
            "start_step": self.global_step,
        })
        # epoch stats
        self.qry_epoch_stats = []

    def on_train_batch_start(self, batch, batch_idx):
        """Log global_step to progress bar instead of GlobalProgressBar
        callback.
        """
        if self.validate:
            if not self.val_with_ft:
                return -1

        if batch_idx == self.mask_steps_per_epoch:
            # Cases:
            # 1. train
            # 2. self.validate + self.val_with_ft
            self.copied.load_state_dict(self.model.state_dict(), strict=True)
            if self.spk_emb_weight_avg:
                with torch.no_grad():
                    self.spk_emb_avg_weight_copied.data.copy_(
                        self.spk_emb_avg_weight_orig.data)
                    self._sample_spk_emb_avg_w_mask()
                    self.spk_emb_avg_weight_mask.detach_()

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.
        Including k-shot support set and q-shot query set forward + evaluate.
        """
        mask_opt, ft_opt = self.optimizers()
        if self.inner_sched:
            mask_sched, ft_sched = self.lr_schedulers()
        elif not self.inner_sched:
            mask_sched, ft_sched = self.lr_schedulers(), None
        batch = batch[0]

        if batch_idx < self.mask_steps_per_epoch:
            if self.validate:
                assert self.val_with_ft
                return None

            mask_batch = batch["mask"]
            if not hasattr(self, "speaker_args"):
                self.register_buffer(
                    "speaker_args", torch.unique(mask_batch["speaker_args"]))

            mask_output = self._train_mask_step(
                mask_batch, batch_idx, mask_opt, mask_sched)

            return {
                "mask": mask_output
            }

        else:
            if self.validate:
                assert self.val_with_ft
                _batch_idx = batch_idx
            else:
                _batch_idx = batch_idx - self.mask_steps_per_epoch
            if self._check_epoch_done(_batch_idx):
                return None

            sup_batch, qry_batch = batch["sup"], batch["qry"]
            qry_batch["speaker_args"] = sup_batch["speaker_args"]

            sup_output, qry_output = self._train_ft_step(
                sup_batch, qry_batch, _batch_idx, ft_opt, ft_sched)

            return {
                "sup": sup_output,
                "qry": qry_output,
            }

    def on_train_epoch_end(self,):
        self.ft_optimizer.__setstate__({'state': defaultdict(dict)})
        if self.inner_sched:
            # self.ft_scheduler.__setstate__({'state': defaultdict(dict)})
            self.ft_scheduler.load_state_dict(self.ft_sched_state_dict)
        self.log("B_id", self.qry_stats[-1]["batch_idx"])

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.logits_and_masks.setup_model(self.model, detach_mask=True)
        self.logits_and_masks.setup_model(self.copied, detach_mask=True)

        if self.spk_emb_weight_avg:
            self._sample_weighted_speaker_emb(
                copied=False, reset_weight_mask=False)
            self._sample_weighted_speaker_emb(
                copied=True, reset_weight_mask=False)
        # Sanity: ??
        # if self.trainer.state.stage == "sanity_check":
        #     return

        output = {}
        df = []

        # 1st-pass: with self.model (before fine-tuning)
        assert not self.model.training
        val_loss, val_output = self.subset_step(batch, "val_start", True)
        output["val_start"] = val_output

        val_start_results = {
            "total_loss": val_loss[0].item(),
            "accent_acc": val_output["accent_acc"].item(),
            "accent_prob": val_output["accent_prob"].mean().item(),
            "speaker_acc": val_output["speaker_acc"].item(),
            "speaker_prob": val_output["speaker_prob"].mean().item(),
        }
        df.append(
            {"stage": "val_start", "batch_idx": batch_idx, **val_start_results})
        self.log("val_m_l", val_start_results["total_loss"])
        self.log("val_m_acc_spk", val_start_results["speaker_acc"])
        self.log("val_m_acc_acnt", val_start_results["accent_acc"])

        # 2nd-pass: with self.copied (after fine-tuning)
        assert not self.copied.training
        val_loss, val_output = self.subset_step(
            batch, "val_end", True, self.copied)
        output["val_end"] = val_output

        val_end_results = {
            "total_loss": val_loss[0].item(),
            "accent_acc": val_output["accent_acc"].item(),
            "accent_prob": val_output["accent_prob"].mean().item(),
            "speaker_acc": val_output["speaker_acc"].item(),
            "speaker_prob": val_output["speaker_prob"].mean().item(),
        }
        df.append(
            {"stage": "val_end", "batch_idx": batch_idx, **val_end_results})
        self.log("val_ft_l", val_end_results["total_loss"])
        self.log("val_ft_acc_spk", val_end_results["speaker_acc"])
        self.log("val_ft_acc_acnt", val_end_results["accent_acc"])

        if self.verbose in {2} and batch_idx == 0:
            self._print_model_stats(substage="")

        return output

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._log_probs_histogram()
        if not self.log_metrics_per_step:
            pre_accent_acc = self.eval_accent_acc.val_start.compute()
            self.eval_accent_acc.val_start.reset()
            post_accent_acc = self.eval_accent_acc.val_end.compute()
            self.eval_accent_acc.val_end.reset()
            pre_speaker_acc = self.eval_speaker_acc.val_start.compute()
            self.eval_speaker_acc.val_start.reset()
            post_speaker_acc = self.eval_speaker_acc.val_end.compute()
            self.eval_speaker_acc.val_end.reset()
            pre_df = {
                "epoch": self.current_epoch,
                "pre_accent_acc": pre_accent_acc.item(),
                "pre_speaker_acc": pre_speaker_acc.item(),
                "post_accent_acc": post_accent_acc.item(),
                "post_speaker_acc": post_speaker_acc.item(),
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

    def _get_mask_stats(self):
        params = []
        for n, p in self.model.named_parameters():
            if n in {"variance_adaptor.pitch_bins",
                     "variance_adaptor.energy_bins"}:
                continue
            elif n[-5:] == "_orig":
                module_name, tensor_name = n[:-5].rsplit('.', 1)
                module = self.model.get_submodule(module_name)
                param = getattr(module, tensor_name)
                if self.mask_mode == "hard":
                    params.append(param != 0)
                else:
                    mask = getattr(module, tensor_name+"_mask")
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
        if self.spk_emb_weight_avg:
            prob = torch.distributions.utils.logits_to_probs(
                self.spk_emb_avg_weight_logits, is_binary=True
            )
            named_probs.append(("spk_emb_avg_weight_logits", prob))
            self.logger.experiment.add_histogram(
                "prob/spk_emb_avg_weight", prob*100, self.current_epoch)
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
        self.print(self.copied.mel_linear.weight)
        self.print((self.copied.mel_linear.weight == 0).sum().item())

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

    def _sample_spk_emb_avg_w_mask(self):
        self.spk_emb_avg_weight_mask = torch.distributions.RelaxedBernoulli(
            torch.tensor([1.0]).to(self.device),
            logits=self.spk_emb_avg_weight_logits
        ).rsample()

    def _sample_weighted_speaker_emb(self, copied=False, reset_weight_mask=False):
        if copied:
            model = self.copied.speaker_emb.model
            spk_emb_avg_weight = self.spk_emb_avg_weight_copied
        else:
            model = self.model.speaker_emb.model
            spk_emb_avg_weight = self.spk_emb_avg_weight_orig
        n_spk = model.weight.shape[0]
        if reset_weight_mask:
            self._sample_spk_emb_avg_w_mask()
        # else:
        #     self.spk_emb_avg_weight_mask.detach_()
        spk_emb_avg_weight = torch.nn.functional.softmax(
            spk_emb_avg_weight, dim=0)
        spk_emb_avg_weight = spk_emb_avg_weight * self.spk_emb_avg_weight_mask
        spk_emb_avg_weight = spk_emb_avg_weight / spk_emb_avg_weight.sum()
        new_spk_emb = spk_emb_avg_weight.view(1,-1) @ model.weight
        new_spk_emb = new_spk_emb.reshape(1, -1).expand(n_spk, -1)
        setattr(model, "weight", new_spk_emb)

    def _train_mask_step(self, mask_batch, batch_idx, mask_opt, mask_sched=None):
        self.logits_and_masks.sample(training=True)
        self.logits_and_masks.setup_model(self.model)

        if self.spk_emb_weight_avg:
            self._sample_weighted_speaker_emb(
                copied=False, reset_weight_mask=True)

        # Forward
        mask_loss, mask_output = self.subset_step(
            mask_batch, "train_mask", False)

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
        self.log("m_en", entropy, sync_dist=True, prog_bar=True)
        self.log("m_en_p", entropy_prob, sync_dist=True, prog_bar=True)
        self.log("m_acc_acnt", mask_output["accent_acc"], sync_dist=True)
        self.log("m_acc_spk", mask_output["speaker_acc"], sync_dist=True)

        # Backward, update model
        loss = mask_loss[0]
        if self.target_sparsity == 0:
            # loss -= 10 * sparsity
            # loss += (10 * (sparsity - 1)) ** 2
            loss += density
        else:
            # loss -= 10 * entropy_prob
            # loss += (10 * (sparsity - self.target_sparsity)) ** 2
            loss += max(density + self.target_sparsity, 1)
        mask_opt.zero_grad()
        self.manual_backward(loss)
        # torch.nn.utils.clip_grad_norm_(self.logits_and_masks.parameters(), 1.0)
        self.clip_gradients(mask_opt, gradient_clip_val=1.0)
        mask_opt.step()
        if mask_sched is not None:
            mask_sched.step()

        return mask_output

    def _train_ft_step(self, sup_batch, qry_batch, _batch_idx, ft_opt,
                       ft_sched=None):
        self.logits_and_masks.setup_model(self.copied, detach_mask=True)

        if self.spk_emb_weight_avg:
            self._sample_weighted_speaker_emb(
                copied=True, reset_weight_mask=False)

        # Forward (sup, qry)
        sup_loss, sup_output = self.subset_step(
            sup_batch, "train_sup", False, self.copied)
        with torch.no_grad():
            qry_loss, qry_output = self.subset_step(
                qry_batch, "train_qry", False, self.copied)

        # Collect query stats
        qry_results = {
            "batch_idx": _batch_idx,
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
                    self.qry_stats[-1]["batch_idx"] = _batch_idx
                    continue
                moving_weight = 0.5
                self.qry_stats[-1][k] *= 1-moving_weight
                self.qry_stats[-1][k] += moving_weight * qry_results[k]
        if self.log_metrics_per_step:
            prev_update = _batch_idx - self.qry_stats[-1]["batch_idx"]
            self.log("prev_update", prev_update, prog_bar=True)

        # Print/log stats
        self.log("ft_sup_l", sup_loss[0], sync_dist=True, prog_bar=True)
        self.log("ft_qry_l", qry_loss[0], sync_dist=True, prog_bar=True)
        self.log("ft_sup_acc_acnt", sup_output["accent_acc"], sync_dist=True)
        self.log("ft_sup_acc_spk", sup_output["speaker_acc"], sync_dist=True)
        self.log("ft_qry_acc_acnt", qry_output["accent_acc"], sync_dist=True)
        self.log("ft_qry_acc_spk", qry_output["speaker_acc"], sync_dist=True)
        self.log(f"ft_ep_{self.current_epoch}/ft_sup_l", sup_loss[0], sync_dist=True)
        self.log(f"ft_ep_{self.current_epoch}/ft_qry_l", qry_loss[0], sync_dist=True)
        self.log(f"ft_ep_{self.current_epoch}/ft_sup_acc_acnt", sup_output["accent_acc"], sync_dist=True)
        self.log(f"ft_ep_{self.current_epoch}/ft_sup_acc_spk", sup_output["speaker_acc"], sync_dist=True)
        self.log(f"ft_ep_{self.current_epoch}/ft_qry_acc_acnt", qry_output["accent_acc"], sync_dist=True)
        self.log(f"ft_ep_{self.current_epoch}/ft_qry_acc_spk", qry_output["speaker_acc"], sync_dist=True)
        if self.verbose in {1, 2}:
            if self.verbose in {2}:
                if _batch_idx == 0:
                    self._print_model_stats(substage="train")
                qry_epoch_df = pd.DataFrame(self.qry_epoch_stats)
                qry_epoch_df = qry_epoch_df.set_index(["batch_idx"])
                qry_epoch_df = qry_epoch_df.to_string().splitlines()
                if _batch_idx == 0:
                    self.print(qry_epoch_df)
                else:
                    self.print(qry_epoch_df[-1])
            if self.trainer.is_last_batch or self._check_epoch_done(_batch_idx+1):
                self.print(pd.DataFrame(self.qry_stats).set_index(["epoch"]))

        # Backward, update model
        if not self._check_epoch_done(_batch_idx+1):
            ft_opt.zero_grad()
            self.manual_backward(sup_loss[0])
            torch.nn.utils.clip_grad_norm_(self.copied.parameters(), 1.0)
            ft_opt.step()
            if self.inner_sched and ft_sched is not None:
                ft_sched.step()

        return sup_output, qry_output
