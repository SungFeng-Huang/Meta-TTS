import os
import re
import pandas as pd
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
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
        # if batch_idx == 0:
        #     pl_module.print(trainer.state.stage, "train_mask")
        #     # print(pl_module.model.mel_linear.weight)
        #     # print((pl_module.model.mel_linear.weight == 0).sum().item())
        #     try:
        #         pl_module.print(pl_module.model.mel_linear.weight)
        #         pl_module.print((pl_module.model.mel_linear.weight == 0).sum().item())
        #         pl_module.print(pl_module.copied.mel_linear.weight)
        #         pl_module.print((pl_module.copied.mel_linear.weight == 0).sum().item())
        #     except:
        #         pl_module.print(pl_module.model.mel_linear.weight_mask)
        #         pl_module.print((pl_module.model.mel_linear.weight_mask == 0).sum().item())
        #         pl_module.print(pl_module.copied.mel_linear.weight_mask)
        #         pl_module.print((pl_module.copied.mel_linear.weight_mask == 0).sum().item())
        # if batch_idx == pl_module.mask_steps_per_epoch:
        #     pl_module.print(trainer.state.stage, "train")
        #     # print(pl_module.model.mel_linear.weight)
        #     # print((pl_module.model.mel_linear.weight == 0).sum().item())
        #     try:
        #         pl_module.print(pl_module.model.mel_linear.weight)
        #         pl_module.print((pl_module.model.mel_linear.weight == 0).sum().item())
        #         pl_module.print(pl_module.copied.mel_linear.weight)
        #         pl_module.print((pl_module.copied.mel_linear.weight == 0).sum().item())
        #     except:
        #         pl_module.print(pl_module.model.mel_linear.weight_mask)
        #         pl_module.print((pl_module.model.mel_linear.weight_mask == 0).sum().item())
        #         pl_module.print(pl_module.copied.mel_linear.weight_mask)
        #         pl_module.print((pl_module.copied.mel_linear.weight_mask == 0).sum().item())

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

        if "mask" in outputs:
            mask_batch = batch[0]["mask"]
            mask_preds = outputs["mask"]["preds"]

            if batch_idx == pl_module.mask_steps_per_epoch:
                # mask_preds
                self.save_mask_wavs(trainer, pl_module, mask_batch, mask_preds, batch_idx)

            # save metrics to csv file
            mask_loss_dict = loss2dict(outputs["mask"]["losses"])
            mask_loss_dict.update({
                k: outputs["mask"][k] for k in outputs["mask"]
                if k not in {"losses", "preds"}
            })
            df = pd.DataFrame([{
                "batch_idx": batch_idx,
                **mask_loss_dict,
            }])
            _csv_dir = os.path.join(self.csv_dir_dict[stage], "mask")
            csv_file_path = os.path.join(_csv_dir, f"epoch={epoch}.csv")
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            df.to_csv(csv_file_path, mode='a', index=False,
                    header=not os.path.exists(csv_file_path))

        else:
            _batch_idx = batch_idx - pl_module.mask_steps_per_epoch
            sup_batch, qry_batch = batch[0]["sup"], batch[0]["qry"]
            sup_preds = outputs["sup"]["preds"]
            qry_preds = outputs["qry"]["preds"]

            if (_batch_idx == 0
                    or trainer.is_last_batch
                    or pl_module._check_epoch_done(_batch_idx+1)):
                # sup_preds
                self.save_sup_wavs(trainer, pl_module, sup_batch, sup_preds, _batch_idx)
                # qry_preds
                self.save_qry_wavs(trainer, pl_module, qry_batch, qry_preds, _batch_idx)

                if (trainer.is_last_batch
                        or pl_module._check_epoch_done(_batch_idx+1)):
                    self.batch_idx = _batch_idx

            # save metrics to csv file
            epoch = trainer.current_epoch
            stage = trainer.state.stage

            qry_loss_dict = loss2dict(outputs["qry"]["losses"])
            qry_loss_dict.update({
                k: outputs["qry"][k] for k in outputs["qry"]
                if k not in {"losses", "preds"}
            })
            df = pd.DataFrame([{
                "batch_idx": _batch_idx,
                **qry_loss_dict,
            }])
            _csv_dir = os.path.join(self.csv_dir_dict[stage], "query")
            csv_file_path = os.path.join(_csv_dir, f"epoch={epoch}.csv")
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            df.to_csv(csv_file_path, mode='a', index=False,
                    header=not os.path.exists(csv_file_path))

            sup_loss_dict = loss2dict(outputs["sup"]["losses"])
            sup_loss_dict.update({
                k: outputs["sup"][k] for k in outputs["sup"]
                if k not in {"losses", "preds"}
            })
            df = pd.DataFrame([{
                "batch_idx": _batch_idx,
                **sup_loss_dict,
            }])
            _csv_dir = os.path.join(self.csv_dir_dict[stage], "support")
            csv_file_path = os.path.join(_csv_dir, f"epoch={epoch}.csv")
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            df.to_csv(csv_file_path, mode='a', index=False,
                    header=not os.path.exists(csv_file_path))

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

        for epoch_start in (True, False):
            tag = "start" if epoch_start else "end"
            _outputs = outputs[f"val_{tag}"]
            preds = _outputs["preds"]
            self.save_val_wavs(trainer, pl_module, batch, preds, epoch_start)

            # csv
            val_ids = batch["ids"]
            accent_prob = _outputs["accent_prob"].tolist()
            accent_acc = _outputs["_accent_acc"].tolist()
            speaker_prob = _outputs["speaker_prob"].tolist()
            speaker_acc = _outputs["_speaker_acc"].tolist()
            df = pd.DataFrame({
                "epoch": [epoch]*len(val_ids),
                "val_ids": val_ids,
                "val_accent_prob": accent_prob,
                "val_accent_acc": accent_acc,
                "val_speaker_prob": speaker_prob,
                "val_speaker_acc": speaker_acc,
            })
            _csv_dir = os.path.join(self.csv_dir_dict[stage], "validate")
            csv_file_path = os.path.join(_csv_dir, f"epoch={epoch}-{tag}.csv")
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            df.to_csv(csv_file_path, mode='a', index=False,
                    header=not os.path.exists(csv_file_path))

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Write my own pruning parser
        pass
