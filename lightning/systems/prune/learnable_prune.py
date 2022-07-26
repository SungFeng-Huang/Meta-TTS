#!/usr/bin/env python3

import pandas as pd
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Any, Optional

import torch
from torch.nn.utils.prune import L1Unstructured, BasePruningMethod
from torchmetrics import Accuracy
from pytorch_lightning.callbacks.pruning import ModelPruning, _MODULE_CONTAINERS
from learn2learn.utils import clone_module, update_module, detach_module, clone_parameters

from lightning.systems.prune.prune_accent import PruneAccentSystem
from lightning.utils import loss2dict
from .utils import global_learnable_unstructured


class LearnablePruneSystem(PruneAccentSystem):
    """
    Inherit from BaseAdaptorSystem for `on_load_checkpoint`.
    """
    def __init__(self,
                 preprocess_config: dict,
                 model_config: dict,
                 algorithm_config: dict,
                 ckpt_path: str,
                 qry_patience: int,
                 log_dir: str = None,
                 result_dir: str = None,
                 accent_ckpt: str = "output/xvec/lightning_logs/version_88/checkpoints/epoch=199-step=164200.ckpt",
                 speaker_ckpt: str= "output/xvec/lightning_logs/version_99/checkpoints/epoch=49-step=377950.ckpt",
                 validate: bool = False,
                 val_with_ft: bool = False,
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

        parameters = ModelPruning.PARAMETER_NAMES
        current_modules = [(n, m) for n, m in self.model.named_modules()
                           if not isinstance(m, _MODULE_CONTAINERS)]
        self.named_parameters_to_prune = [
            (n, m, p) for p in parameters for n, m in current_modules
            if getattr(m, p, None) is not None
        ]
        self.parameters_to_prune = [
            (m, p) for _, m, p in self.named_parameters_to_prune]

        global_learnable_unstructured(
            self.parameters_to_prune,
            L1Unstructured,
            amount=0.1,
        )
        for n, p in self.model.named_parameters():
            if "_orig" in n:
                p.requires_grad_(False)
            elif "_logits" in n:
                p.requires_grad_(True)

        self._on_pre_self_copy()
        self.copied = deepcopy(self.model)
        self._on_post_self_copy()

        # turn to false after debug
        self.copied.load_state_dict(self.model.state_dict(), strict=True)

        self.automatic_optimization = False

        self.validate = validate
        self.val_with_ft = val_with_ft

    def _on_pre_self_copy(self):
        r"""Remove buffers w/ grad_fn before deepcopy.
        `deepcopy` only support leaf tensors (Parameters).
        """
        for module, name in self.parameters_to_prune:
            # parameters
            assert hasattr(module, name + "_orig")
            if module.training:
                assert hasattr(module, name + "_logits")
            # buffers w/ grad_fn
            assert hasattr(module, name + "_mask")
            assert hasattr(module, name)

            # detach mask
            mask = getattr(module, name + "_mask").detach()
            setattr(module, name + "_mask", mask)
            # remove intermediate tensors for deepcopy
            delattr(module, name)

    def _on_post_self_copy(self):
        r"""Reset forward_pre_hooks to normal BasePruningMethod.
        Set param_orig.requires_grad = True.
        Set param_mask.requires_grad = False.
        Remove param_logits.
        Set model.train().
        """
        model = self.copied
        for module_name, _, param_name in self.named_parameters_to_prune:
            module = model.get_submodule(module_name)
            name = param_name
            # detached parameters
            assert hasattr(module, name + "_orig")
            assert hasattr(module, name + "_logits")
            # buffers w/o grad_fn, requires_grad = False
            assert hasattr(module, name + "_mask")

            getattr(module, name + "_orig").requires_grad_(True)

            # probably not necessary
            getattr(module, name + "_mask").requires_grad_(False)

            # getattr(module, name + "_logits").requires_grad_(False)
            # delattr(module, name + "_logits")

            # same hook with "_update_mask"=False to prevent resampling
            setattr(module, "_update_mask", False)
        model.train()

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.mask_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.ft_optimizer = torch.optim.Adam(self.copied.parameters(), lr=3e-5)
        # self.ft_optimizer = torch.optim.SGD(self.copied.parameters(), lr=1e-3)
        # self.opt_init_state = self.ft_optimizer.state_dict

        return self.mask_optimizer, self.ft_optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = self.model.state_dict()
        copied_state_dict = self.copied.state_dict()
        mask_state_dict = {}
        mask_state_dict = {}
        logits_state_dict = {}

        map_pruned_params = {k.replace("_mask", "")
                             for k in state_dict.keys() if k.endswith("_mask")}
        for tensor_name in map_pruned_params:
            orig = state_dict.pop(tensor_name + "_orig")
            mask = state_dict.pop(tensor_name + "_mask")
            logits = state_dict.pop(tensor_name + "_logits")
            # make weights permanent
            state_dict[tensor_name] = mask.to(dtype=orig.dtype) * orig
            mask_state_dict[tensor_name] = mask
            logits_state_dict[tensor_name] = logits

            copied_state_dict.pop(tensor_name)
            copied_state_dict.pop(tensor_name + "_mask")
            copied_state_dict.pop(tensor_name + "_logits")
        checkpoint["state_dict"] = state_dict
        checkpoint["mask_state_dict"] = mask_state_dict
        checkpoint["logits_state_dict"] = logits_state_dict
        checkpoint["copied_state_dict"] = copied_state_dict

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"]
        mask_state_dict = checkpoint["mask_state_dict"]
        logits_state_dict = checkpoint["logits_state_dict"]
        copied_state_dict = checkpoint["copied_state_dict"]
        model_state_dict = {
            **{f"model.{k}": v for k, v in state_dict.items()},
            **{f"model.{k}_mask": v for k, v in mask_state_dict.items()},
            **{f"model.{k}_logits": v for k, v in logits_state_dict.items()},
            **{f"copied.{k}": v for k, v in copied_state_dict.items()},
            **{f"copied.{k}_mask": v for k, v in mask_state_dict.items()},
            **{f"copied.{k}_logits": v for k, v in logits_state_dict.items()},
        }
        checkpoint["state_dict"] = model_state_dict
        del checkpoint["mask_state_dict"]
        del checkpoint["logits_state_dict"]
        del checkpoint["copied_state_dict"]

    def on_fit_start(self,):
        super().on_fit_start()
        self.eval_accent_acc.update({
            "train_mask": Accuracy(ignore_index=-100).to(self.device),
        })
        self.eval_speaker_acc.update({
            "train_mask": Accuracy(ignore_index=-100).to(self.device),
        })

    def on_train_batch_start(self, batch, batch_idx):
        """Log global_step to progress bar instead of GlobalProgressBar
        callback.
        """
        if self.validate:
            if not self.val_with_ft:
                return -1
        self.log("batch_idx", batch_idx, prog_bar=True)
        if batch_idx == 200:
            # Cases:
            # 1. train
            # 2. self.validate + self.val_with_ft
            self.copied.load_state_dict(self.model.state_dict(), strict=True)

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.
        Including k-shot support set and q-shot query set forward + evaluate.
        """
        mask_opt, ft_opt = self.optimizers()

        if batch_idx < 200:
            if self.validate:
                assert self.val_with_ft
                return None
            batch = batch[0]
            mask_batch = batch["mask"]

            mask_loss, mask_output = self.subset_step(
                mask_batch, "train_mask", False)
            self.log("loss", mask_loss[0], sync_dist=True, prog_bar=True)

            curr_stats = []
            for n, m, p in self.named_parameters_to_prune:
                mask = getattr(m, p + "_mask")
                num = mask.numel()
                zeros = num - mask.sum()
                curr_stats.append((zeros, mask.numel()))
            total_params = sum(param_size for _, param_size in curr_stats)
            total_zeros = sum(zeros for zeros, _ in curr_stats)
            self.print(batch_idx, f"Pruned: {total_zeros}/{total_params}",
                       f"({total_zeros/total_params:.2%})")
            self.log("sparsity", total_zeros/total_params, sync_dist=True,
                     prog_bar= True)
            self.qry_stats[-1]["sparsity"] = (total_zeros/total_params).item()*100

            mask_opt.zero_grad()
            self.manual_backward(mask_loss[0] - 100*total_zeros/total_params)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            mask_opt.step()

            return {
                "mask": mask_output
            }

        else:
            if self.validate:
                assert self.val_with_ft
                _batch_idx = batch_idx
            else:
                _batch_idx = batch_idx-200
            if self._check_epoch_done(_batch_idx):
                return None
            batch = batch[0]
            sup_batch, qry_batch = batch["sup"], batch["qry"]
            qry_batch["speaker_args"] = sup_batch["speaker_args"]

            sup_loss, sup_output = self.subset_step(
                sup_batch, "train_sup", False, self.copied)
            self.log("loss", sup_loss[0], sync_dist=True, prog_bar=True)

            with torch.no_grad():
                qry_loss, qry_output = self.subset_step(
                    qry_batch, "train_qry", True, self.copied)

            qry_results = {
                "batch_idx": _batch_idx,
                "total_loss": qry_loss[0].item(),
                **{
                    k: v for k, v in qry_output.items()
                    if k not in {"losses", "preds"}
                },
            }
            if "batch_idx" not in self.qry_stats[-1]:
                self.qry_stats[-1].update(qry_results)
            else:
                if (
                    qry_results["accent_acc"] >= self.qry_stats[-1]["accent_acc"]
                    and qry_results["accent_prob"] >= self.qry_stats[-1]["accent_prob"]
                    and qry_results["speaker_acc"] >= self.qry_stats[-1]["speaker_acc"]
                    and qry_results["speaker_prob"] >= self.qry_stats[-1]["speaker_prob"]
                ) and (
                    qry_results["accent_acc"] > self.qry_stats[-1]["accent_acc"]
                    or qry_results["accent_prob"] > self.qry_stats[-1]["accent_prob"]
                    or qry_results["speaker_acc"] > self.qry_stats[-1]["speaker_acc"]
                    or qry_results["speaker_prob"] > self.qry_stats[-1]["speaker_prob"]
                ):
                    self.qry_stats[-1]["batch_idx"] = _batch_idx
                for k in qry_results:
                    if k in {"batch_idx"}:
                        continue
                    moving_weight = 0.5
                    self.qry_stats[-1][k] *= 1-moving_weight
                    self.qry_stats[-1][k] += moving_weight * qry_results[k]

            if self.log_metrics_per_step:
                prev_update = _batch_idx - self.qry_stats[-1]["batch_idx"]
                self.log("prev_update", prev_update, prog_bar=True)
            self.print(pd.DataFrame([qry_results]))
            self.print(pd.DataFrame(self.qry_stats))

            if not self._check_epoch_done(_batch_idx+1):
                ft_opt.zero_grad()
                self.manual_backward(sup_loss[0])
                torch.nn.utils.clip_grad_norm_(self.copied.parameters(), 1.0)
                ft_opt.step()

            return {
                "sup": sup_output,
                "qry": qry_output,
            }

    def on_train_epoch_end(self,):
        self.ft_optimizer.__setstate__({'state': defaultdict(dict)})

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Sanity: ??
        # if self.trainer.state.stage == "sanity_check":
        #     return

        output = {}
        df = []

        # 1st-pass: with self.model (before fine-tuning)
        assert not self.model.training
        val_loss, val_output = self.subset_step(batch, "val_start", True)
        val_start_results = {
            "total_loss": val_loss[0].item(),
            "accent_acc": val_output["accent_acc"].item(),
            "accent_prob": val_output["accent_prob"].mean().item(),
            "speaker_acc": val_output["speaker_acc"].item(),
            "speaker_prob": val_output["speaker_prob"].mean().item(),
        }
        # self.print("val_start", batch_idx, val_results)
        output["val_start"] = val_output
        df.append(
            {"stage": "val_start", "batch_idx": batch_idx, **val_start_results})

        # 2nd-pass: with self.copied (after fine-tuning)
        assert not self.copied.training
        val_loss, val_output = self.subset_step(
            batch, "val_end", True, self.copied)
        val_end_results = {
            "total_loss": val_loss[0].item(),
            "accent_acc": val_output["accent_acc"].item(),
            "accent_prob": val_output["accent_prob"].mean().item(),
            "speaker_acc": val_output["speaker_acc"].item(),
            "speaker_prob": val_output["speaker_prob"].mean().item(),
        }
        # self.print("val_end", batch_idx, val_results)
        output["val_end"] = val_output
        df.append(
            {"stage": "val_end", "batch_idx": batch_idx, **val_end_results})

        if batch_idx == 0:
            self.print(self.trainer.state.stage)
            self.print(self.model.mel_linear.weight)
            self.print((self.model.mel_linear.weight == 0).sum().item())
            self.print(self.copied.mel_linear.weight)
            self.print((self.copied.mel_linear.weight == 0).sum().item())
        self.print(pd.DataFrame(df))

        return output
