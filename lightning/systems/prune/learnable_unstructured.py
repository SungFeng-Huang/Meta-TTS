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

from lightning.systems.prune.prune_accent import PruneAccentSystem
from lightning.utils import loss2dict
from .utils import global_learnable_unstructured


class LearnableUnstructuredPruneSystem(PruneAccentSystem):
    """
    Inherit from BaseAdaptorSystem for `on_load_checkpoint`.
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

        self.target_sparsity = target_sparsity
        # init_sparsity = 0.5 if target_sparsity == 0 else target_sparsity
        init_sparsity = 0.1
        # RandomUnstructured might cause error forwarding
        global_learnable_unstructured(
            self.parameters_to_prune,
            L1Unstructured,
            amount=init_sparsity,
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
        self.mask_steps_per_epoch = mask_steps_per_epoch
        self.verbose = verbose

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

            try:
                copied_state_dict.pop(tensor_name)
            except:
                pass
            copied_state_dict.pop(tensor_name + "_mask")
            copied_state_dict.pop(tensor_name + "_logits")
        checkpoint["state_dict"] = state_dict
        checkpoint["mask_state_dict"] = mask_state_dict
        checkpoint["logits_state_dict"] = logits_state_dict
        checkpoint["copied_state_dict"] = copied_state_dict
        checkpoint["speaker_args"] = self.speaker_args

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
            # "speaker_args": checkpoint["speaker_args"],
        }
        self.register_buffer("speaker_args", checkpoint["speaker_args"])
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
        self.val_stats = {}

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
        self.log("batch_idx", batch_idx, prog_bar=True)

        if batch_idx == self.mask_steps_per_epoch:
            # Cases:
            # 1. train
            # 2. self.validate + self.val_with_ft
            self.copied.load_state_dict(self.model.state_dict(), strict=True)

    def _get_mask_stats(self):
        masks = torch.nn.utils.parameters_to_vector(
            [
                getattr(module, name + "_mask")
                for (module, name) in self.parameters_to_prune
            ]
        )
        n_params = masks.numel()
        n_ones = masks.sum()
        n_zeros = n_params - n_ones
        density = n_ones / n_params
        return n_params, n_zeros, density

    def _get_mask_entropy_stats(self):
        logits = torch.nn.utils.parameters_to_vector(
            [
                getattr(module, name + "_logits")
                for (module, name) in self.parameters_to_prune
            ]
        )
        probs = torch.distributions.utils.logits_to_probs(logits,
                                                            is_binary=True)
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
        self.log("sparsity", sparsity, sync_dist=True, prog_bar=True)
        self.log("entropy", entropy, sync_dist=True, prog_bar=True)
        self.log("ent_prob", entropy_prob, sync_dist=True, prog_bar=True)
        self.qry_stats[-1]["sparsity"] = sparsity.item()*100
        self.qry_stats[-1]["entropy_prob"] = entropy_prob.item()*100

    def _train_mask_step(self, mask_batch, batch_idx, mask_opt):
        # Forward
        mask_loss, mask_output = self.subset_step(
            mask_batch, "train_mask", False)
        self.log("mask_loss", mask_loss[0], sync_dist=True, prog_bar=True)

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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        mask_opt.step()

        return mask_output

    def _train_ft_step(self, sup_batch, qry_batch, _batch_idx, ft_opt):
        # Forward (sup, qry)
        sup_loss, sup_output = self.subset_step(
            sup_batch, "train_sup", False, self.copied)
        self.log("loss", sup_loss[0], sync_dist=True, prog_bar=True)
        with torch.no_grad():
            qry_loss, qry_output = self.subset_step(
                qry_batch, "train_qry", True, self.copied)

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

        return sup_output, qry_output

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.
        Including k-shot support set and q-shot query set forward + evaluate.
        """
        mask_opt, ft_opt = self.optimizers()
        batch = batch[0]

        if batch_idx < self.mask_steps_per_epoch:
            if self.validate:
                assert self.val_with_ft
                return None

            mask_batch = batch["mask"]
            if not hasattr(self, "speaker_args"):
                self.register_buffer(
                    "speaker_args", torch.unique(mask_batch["speaker_args"]))

            mask_output = self._train_mask_step(mask_batch, batch_idx, mask_opt)

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
                sup_batch, qry_batch, _batch_idx, ft_opt)

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

        if self.verbose in {2} and batch_idx == 0:
            self._print_model_stats(substage="")
        # self.print(pd.DataFrame(df))

        return output

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
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
