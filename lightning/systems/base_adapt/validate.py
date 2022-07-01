#!/usr/bin/env python3

import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
# from learn2learn.algorithms import MAML
from torchmetrics import Accuracy

from lightning.systems.base_adapt.base import BaseAdaptorSystem
from lightning.callbacks.saver import ValidateSaver
# from lightning.metrics import XvecAccentAccuracy
from lightning.model.xvec import XvecTDNN
from lightning.utils import loss2dict


VERBOSE = False

class BaseAdaptorValidateSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, preprocess_config, model_config, train_config,
                 algorithm_config, log_dir=None, result_dir=None):
        super().__init__(preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir)

        assert "saving_steps" in self.algorithm_config["adapt"]["test"]
        self.saving_steps = self.algorithm_config["adapt"]["test"]["saving_steps"]

    def configure_callbacks(self):
        # Save figures/audios/csvs
        saver = ValidateSaver(self.preprocess_config, self.log_dir, self.result_dir)
        self.saver = saver

        callbacks = [saver]
        return callbacks

    def on_validation_start(self,):
        self.mem_stats = torch.cuda.memory_stats(self.device)
        self.csv_path = os.path.join(self.log_dir, "csv",
                                     f"validate-{self.global_step}.csv")
        os.makedirs(os.path.join(self.log_dir, "csv"))
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

        # initialize eval_model here to avoid checkpoint loading error
        xvec_ckpt_path = "output/xvec/lightning_logs/version_88/checkpoints/epoch=199-step=164200.ckpt"
        self.eval_model = XvecTDNN.load_from_checkpoint(xvec_ckpt_path).to(self.device)
        self.eval_model.freeze()
        self.eval_model.eval()
        self.eval_acc = torch.nn.ModuleDict({
            "recon": Accuracy(ignore_index=-100),
            **{f"step_{ft_step}": Accuracy(ignore_index=-100)
               for ft_step in [0] + self.saving_steps},
        }).to(self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ LightningModule interface.
        Operates on a single batch of data from the validation set.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
        outputs = self._test_step(batch, batch_idx, dataloader_idx)

        st = time.time()
        if torch.count_nonzero(outputs["_batch"]["accents"] >= 0) > 0:
            mask = outputs["_batch"]["accents"] >= 0
            with torch.no_grad():
                recon_acc = self.eval_model(
                    outputs["step_0"]["recon"]["output"][1].transpose(1, 2))
            recon_acc_prob = F.softmax(recon_acc, dim=-1)[
                torch.arange(recon_acc.shape[0]).to(self.device)[mask],
                outputs["_batch"]["accents"][mask]]
            self.log("recon_acc_prob", recon_acc_prob)
            self.eval_acc["recon"](recon_acc, outputs["_batch"]["accents"])
            self.log("recon_acc", self.eval_acc["recon"])
        for ft_step in [0] + self.saving_steps:
            step = f"step_{ft_step}"
            loss_dict = loss2dict(outputs[f"step_{ft_step}"]["recon"]["losses"])
            data = {
                "global_step": self.global_step,
                "ft_step": ft_step,
                "task_id": outputs["task_id"],
                "dset_name": outputs["task_id"].rsplit('/', 1)[0],
                "dataloader_idx": dataloader_idx,
                "batch_idx": batch_idx,
                "accent_acc": 0,
                "accent_prob": 0,
                **loss_dict,
            }
            if torch.count_nonzero(outputs["_batch"]["accents"] >= 0) > 0:
                mask = outputs["_batch"]["accents"] >= 0
                step_acc = self.eval_model(
                    outputs[step]["synth"]["output"][1].transpose(1, 2))
                step_acc_prob = F.softmax(step_acc, dim=-1)[
                    torch.arange(step_acc.shape[0]).to(self.device)[mask],
                    outputs["_batch"]["accents"][mask]]
                self.log(f"{step}_acc_prob", step_acc_prob)
                accent_acc = self.eval_acc[step](step_acc, outputs["_batch"]["accents"])
                self.log(f"{step}_acc", self.eval_acc[step])
                data["accent_acc"] = accent_acc.item()
                data["accent_prob"] = step_acc_prob.item()
            df = pd.DataFrame([data])
            df.to_csv(self.csv_path, index=False, mode='a', header=not
                      os.path.exists(self.csv_path))
        self._print_mem_diff(st, "Log_dict")

        st = time.time()
        torch.cuda.empty_cache()
        self._print_mem_diff(st, "Empty cache")
        return outputs

    def on_test_start(self):
        if self.algorithm_config["adapt"]["speaker_emb"] == "table":
            if self.local_rank == 0:
                print("Testing speaker emb")
                print("Before:")
                print(self.model.speaker_emb.model.weight[-39:])

            if (self.preprocess_config["dataset"] == "LibriTTS"
                    and self.algorithm_config["adapt"]["test"].get("avg_train_spk_emb", False)):

                with torch.no_grad():
                    self.model.speaker_emb.model.weight[-39:] = \
                        self.model.speaker_emb.model.weight[:247].mean(dim=0)

            if self.local_rank == 0:
                print("After:")
                print(self.model.speaker_emb.model.weight[-39:])
                print()

    def _test_step(self, batch, batch_idx, dataloader_idx):
        adapt_steps = [step_ - _step for _step, step_ in
                       zip([0] + self.saving_steps, self.saving_steps)]
        # sup_ids = batch[0][0][0][0]
        # qry_ids = batch[0][1][0][0]
        sup_ids = batch[0][0][0]["ids"]
        qry_ids = batch[0][1][0]["ids"]
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
        self._print_mem_diff(st, "Adapt 0")

        st = time.time()
        torch.cuda.empty_cache()
        self._print_mem_diff(st, "Empty cache")

        # Adapt
        for ft_step, adapt_step in zip(self.saving_steps, adapt_steps):
            st = time.time()
            learner, val_loss, predictions = self.meta_learn(
                batch, adapt_step, learner=learner, train=False,
                synth=True, recon=False,
            )
            outputs[f"step_{ft_step}"] = {
                "recon": {"losses": val_loss},
                "synth": {"output": predictions},
            }
            self._print_mem_diff(st, f"Adapt ~{ft_step} ({adapt_step})")

            st = time.time()
            torch.cuda.empty_cache()
            self._print_mem_diff(st, "Empty cache")

        del learner

        return outputs

