import os
import re
import pandas as pd
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.io import wavfile

from typing import Any, Mapping, Literal, Dict, Optional
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.pruning import ModelPruning, _MODULE_CONTAINERS

from lightning.utils import loss2dict
# from lightning.callbacks.utils import recon_samples, synth_samples
from .prune_accent import PruneAccentSaver


class LearnablePruneSaver(PruneAccentSaver):
    def save_mask_wavs(self,
                      trainer: Trainer,
                      pl_module: LightningModule,
                      mask_batch: Any,
                      mask_preds: Any,
                      batch_idx: int) -> None:
        return self.save_wavs(
            "mask", trainer, pl_module, mask_batch, mask_preds, batch_idx)

    def on_train_batch_start(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           batch: Any,
                           batch_idx: int):
        pass

    def on_train_batch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           outputs: Mapping,
                           batch: Any,
                           batch_idx: int):
        if outputs is None or len(outputs) == 0:
            return

        epoch = trainer.current_epoch
        stage = trainer.state.stage

        self.vocoder.to(pl_module.device)
        self.vocoder.eval()

        if isinstance(batch, dict):
            new_batch = [batch["ft"][0]]
            if "mask" in batch:
                new_batch[0]["mask"] = batch["mask"]
            batch = new_batch

        for key in outputs: # mask, sup, qry
            _batch = batch[0][key]
            _outputs = outputs[key]
            _preds = _outputs["preds"]

            if trainer.is_last_batch:
                # mask_preds
                self.save_wavs(
                    key, trainer, pl_module, _batch, _preds, batch_idx)

            # save metrics to csv file
            _loss_dict = loss2dict(_outputs["losses"])
            _loss_dict.update({
                k: v for k, v in _outputs.items()
                if k not in {"losses", "preds"}
            })
            df = pd.DataFrame([{
                "batch_idx": batch_idx,
                **_loss_dict,
            }])
            _csv_dir = os.path.join(self.csv_dir_dict[stage], key)
            csv_file_path = os.path.join(_csv_dir, f"epoch={epoch}.csv")
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            df.to_csv(csv_file_path, mode='a', index=False,
                    header=not os.path.exists(csv_file_path))

            # if (batch_idx == 0
            #         or trainer.is_last_batch
            #         or pl_module._check_epoch_done(batch_idx+1)):
            #     if (trainer.is_last_batch
            #             or pl_module._check_epoch_done(batch_idx+1)):
            #         self.batch_idx = batch_idx

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Mapping,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int):
        if outputs is None or len(outputs) == 0:
            return

        self.vocoder.to(pl_module.device)
        self.vocoder.eval()

        epoch = trainer.current_epoch
        stage = trainer.state.stage

        _outputs = outputs[f"val"]
        preds = _outputs["preds"]
        self.save_wavs(
            "validate", trainer, pl_module, batch, preds, batch_idx=0)

        # csv
        df = pd.DataFrame({
            "epoch": [epoch]*len(batch["ids"]),
            "sparsity": _outputs["sparsity"],
            "val_ids": batch["ids"],
            "val_accent_prob": _outputs["accent_prob"].tolist(),
            "val_accent_acc": _outputs["_accent_acc"].tolist(),
            "val_speaker_prob": _outputs["speaker_prob"].tolist(),
            "val_speaker_acc": _outputs["_speaker_acc"].tolist(),
        })
        _csv_dir = os.path.join(self.csv_dir_dict[stage], "validate")
        csv_file_path = os.path.join(_csv_dir, f"epoch={epoch}.csv")
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df.to_csv(csv_file_path, mode='a', index=False,
                header=not os.path.exists(csv_file_path))

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for n, p in pl_module.model.named_parameters():
            if n in {"variance_adaptor.pitch_bins",
                     "variance_adaptor.energy_bins"}:
                continue
            elif n[-5:] == "_orig":
                module_name, tensor_name = n[:-5].rsplit('.', 1)
                module = pl_module.model.get_submodule(module_name)
                mask = getattr(module, tensor_name+"_mask")
                if n[:-5] == "speaker_emb.model.weight":
                    mask = mask[0]    # only 1 spk

                fig_name = '/'.join(("mask_"+n[:-5]).split('.'))
                mask = mask.detach().cpu()
                if mask.ndim == 3:
                    if n in {"encoder.position_enc_orig",
                            "decoder.position_enc_orig"}:
                        mask = mask[0]
                    else:
                        mask = mask[..., 0]
                elif mask.ndim == 1:
                    mask = mask[None]
                elif mask.ndim != 2:
                    raise f"shape: {mask.shape}"

                # sns.clustermap(mask,
                #                row_cluster=(mask.shape[0]!=1),
                #                col_cluster=(mask.shape[1]!=1),
                in_dense = (mask.sum(dim=0) != 0).sum().item() / mask.shape[1]
                out_dense = (mask.sum(dim=1) != 0).sum().item() / mask.shape[0]
                sns.heatmap(mask, vmin=0, vmax=1, square=True).set(
                    title=f"in: {1-in_dense:.4f}; out: {1-out_dense:.4f}; total: {1-in_dense*out_dense:.4f}"
                )

                self.log_figure(
                    pl_module.logger,
                    fig_name,
                    plt.gcf(),
                    pl_module.global_step,
                )
                plt.close()

        pass

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Write my own pruning parser
        pass

    def _log_probs_histogram(self):
        named_probs = []
        import torch
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
                self.spk_emb_w_avg_subset_logits, is_binary=True
            )
            named_probs.append(("spk_emb_w_avg_subset_logits", prob))
            self.logger.experiment.add_histogram(
                "prob/spk_emb_w_avg_subset", prob*100, self.current_epoch)
        probs = torch.nn.utils.parameters_to_vector([p for _, p in named_probs])
        self.logger.experiment.add_histogram("probs", probs*100, self.current_epoch)
