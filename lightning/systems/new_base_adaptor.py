#!/usr/bin/env python3

import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor, ModelCheckpoint, RichProgressBar
import learn2learn as l2l
# from learn2learn.algorithms import MAML
from learn2learn.utils import detach_module
from resemblyzer import VoiceEncoder
from torchmetrics import Accuracy

# from lightning.systems.system import System
from lightning.systems.utils import Task, MAML
from lightning.metrics import XvecAccentAccuracy
from lightning.model.xvec import XvecTDNN
from lightning.model import FastSpeech2Loss, FastSpeech2
# from lightning.callbacks import GlobalProgressBar, Saver
from lightning.callbacks import Saver
from lightning.optimizer import get_optimizer
from lightning.scheduler import get_scheduler
from lightning.utils import LightningMelGAN, loss2dict
from utils.tools import expand, plot_mel, get_mask_from_lengths


VERBOSE = False

class BaseAdaptorSystem(pl.LightningModule):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
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
        # self.learner.inner_freeze()
        # for module in self.algorithm_config["adapt"]["modules"]:
            # self.learner.inner_unfreeze(module)

        self.adaptation_steps      = self.algorithm_config["adapt"]["train"]["steps"]
        self.test_adaptation_steps = self.algorithm_config["adapt"]["test"]["steps"]
        assert self.test_adaptation_steps % self.adaptation_steps == 0

        self.d_target = False
        self.reference_prosody = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_idx, train=True):
        output = self(**batch)
        loss = self.loss_func(batch, output)
        return loss, output

    def on_train_epoch_start(self):
        self.train_loss_dicts = []

    def on_train_batch_start(self, batch, batch_idx):
        # self.print("start", self.global_step, batch_idx)
        pass

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        """Not used. Don't use."""
        loss, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': loss[0], 'losses': loss, 'output': output, '_batch': batch}

    def on_train_epoch_end(self):
        key_map = {
            "Train/Total Loss": "total_loss",
            "Train/Mel Loss": "mel_loss",
            "Train/Mel-Postnet Loss": "_mel_loss",
            "Train/Duration Loss": "d_loss",
            "Train/Pitch Loss": "p_loss",
            "Train/Energy Loss": "e_loss",
        }
        loss_dict = {"step": self.global_step}
        for k in key_map:
            loss_dict[key_map[k]] = self.trainer.callback_metrics[k].item()

        self.train_loss_dicts.append(loss_dict)
        df = pd.DataFrame(self.train_loss_dicts).set_index("step")
        if (self.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0:
            self.print(df.to_string(header=True, index=True))
        else:
            self.print(df.to_string(header=True, index=True).split('\n')[-1])

    def on_validation_start(self,):
        self.mem_stats = torch.cuda.memory_stats(self.device)
        self.csv_path = os.path.join(self.log_dir, "csv",
                                     f"validate-{self.global_step}.csv")
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

        assert "saving_steps" in self.algorithm_config["adapt"]["test"]
        self.saving_steps = self.algorithm_config["adapt"]["test"]["saving_steps"]

        xvec_ckpt_path = "output/xvec/lightning_logs/version_88/checkpoints/epoch=199-step=164200.ckpt"
        self.eval_model = XvecTDNN.load_from_checkpoint(xvec_ckpt_path).to(self.device)
        self.eval_model.freeze()
        self.eval_model.eval()
        self.eval_acc = torch.nn.ModuleDict({
            "recon": XvecAccentAccuracy(xvec_ckpt_path),
            **{f"step_{ft_step}": XvecAccentAccuracy(xvec_ckpt_path)
               for ft_step in [0] + self.saving_steps},
        })
        # self.eval_acc = torch.nn.ModuleDict({
        #     "recon": XvecAccentAccuracy(xvec_ckpt_path),
        #     **{f"step_{ft_step}": XvecAccentAccuracy(xvec_ckpt_path)
        #        for ft_step in [0] + self.saving_steps},
        # })
        # for eval_acc in self.eval_acc.values():
        #     eval_acc.add_model(self.eval_model)
        self.eval_acc = torch.nn.ModuleDict({
            "recon": Accuracy(),
            **{f"step_{ft_step}": Accuracy()
               for ft_step in [0] + self.saving_steps},
        })
        for eval_acc in self.eval_acc.values():
            eval_acc.to(self.device)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ LightningModule interface.
        Called in the validation loop before anything happens for that batch.
        """
        self._on_meta_batch_start(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ LightningModule interface.
        Operates on a single batch of data from the validation set.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
        if self.trainer.state.fn == "fit":
            return self._val_step_for_fit(batch, batch_idx, dataloader_idx)
        elif self.trainer.state.fn == "validate":
            return self._val_step_for_validate(batch, batch_idx, dataloader_idx)

    def validation_epoch_end(self, outputs):
        pass

    def on_validation_end(self):
        """ LightningModule interface.
        Called at the end of validation.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
        if self.trainer.state.fn == "fit":
            return self._on_val_end_for_fit()
        elif self.trainer.state.fn == "validate":
            for eval_acc in self.eval_acc.values():
                # eval_acc.reset()
                pass

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

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        """ LightningModule interface.
        Called in the test loop before anything happens for that batch.
        """
        self._on_meta_batch_start(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        all_outputs = []
        qry_batch = batch[0][1][0]
        if self.algorithm_config["adapt"]["test"].get("1-shot", False):
            task = Task(sup_data=batch[0][0][0], batch_size=1, shuffle=False)
            for sup_batch in iter(task):
                mini_batch = [([sup_batch], [qry_batch])]
                outputs = self._test_step(mini_batch, batch_idx, dataloader_idx)
                all_outputs.append(outputs)
        else:
            outputs = self._test_step(batch, batch_idx, dataloader_idx)
            all_outputs.append(outputs)
        torch.distributed.barrier()

        return all_outputs

    def configure_callbacks(self):
        # Checkpoint saver
        save_step = self.train_config["step"]["save_step"]
        checkpoint = ModelCheckpoint(
            monitor="Train/Total Loss", mode="min", # monitor not used
            every_n_train_steps=save_step, save_top_k=-1, save_last=True,
        )

        # Progress bars (step/epoch)
        # outer_bar = GlobalProgressBar(process_position=1)
        # inner_bar = ProgressBar(process_position=1) # would * 2
        # inner_bar = RichProgressBar()

        # Monitor learning rate / gpu stats
        lr_monitor = LearningRateMonitor()
        # gpu_monitor = GPUStatsMonitor(
        #     memory_utilization=True, gpu_utilization=True, intra_step_time=True, inter_step_time=True
        # )

        # Save figures/audios/csvs
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        self.saver = saver

        # callbacks = [checkpoint, outer_bar, lr_monitor, gpu_monitor, saver]
        # callbacks = [checkpoint, lr_monitor, gpu_monitor, saver]
        callbacks = [checkpoint, lr_monitor, saver]
        # callbacks = []
        return callbacks

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = get_optimizer(self.model, self.model_config, self.train_config)

        self.scheduler = {
            "scheduler": get_scheduler(self.optimizer, self.train_config),
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            'monitor': self.default_monitor,
        }

        return [self.optimizer], [self.scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        changes = {"skip": [], "drop": [], "replace": [], "miss": []}

        if 'model.speaker_emb.weight' in state_dict:
            # Compatiable with old version
            assert "model.speaker_emb.model.weight" in model_state_dict
            assert "model.speaker_emb.model.weight" not in state_dict
            state_dict["model.speaker_emb.model.weight"] = state_dict.pop("model.speaker_emb.weight")
            changes["replace"].append(["model.speaker_emb.weight", "model.speaker_emb.model.weight"])
            is_changed = True

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if k == "model.speaker_emb.model.weight":
                        # train-clean-100: 247 (spk_id 0 ~ 246)
                        # train-clean-360: 904 (spk_id 247 ~ 1150)
                        # train-other-500: 1160 (spk_id 1151 ~ 2310)
                        # dev-clean: 40 (spk_id 2311 ~ 2350)
                        # test-clean: 39 (spk_id 2351 ~ 2389)
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
                                if self.local_rank == 0:
                                    print("Average training speaker emb as testing emb")
                    # Other testing corpus with different # of spks: re-initialize
                    changes["skip"].append([
                        k, model_state_dict[k].shape, state_dict[k].shape
                    ])
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                changes["drop"].append(k)
                is_changed = True

        for k in model_state_dict:
            if k not in state_dict:
                changes["miss"].append(k)
                is_changed = True

        if self.local_rank == 0:
            print()
            for replaced in changes["replace"]:
                before, after = replaced
                print(f"Replace: {before}\n\t-> {after}")
            for skipped in changes["skip"]:
                k, required_shape, loaded_shape = skipped
                print(f"Skip parameter: {k}, \n" \
                           + f"\trequired shape: {required_shape}, " \
                           + f"loaded shape: {loaded_shape}")
            for dropped in changes["drop"]:
                del state_dict[dropped]
            for dropped in set(['.'.join(k.split('.')[:2]) for k in changes["drop"]]):
                print(f"Dropping parameter: {dropped}")
            for missed in set(['.'.join(k.split('.')[:2]) for k in changes["miss"]]):
                print(f"Missing parameter: {missed}")
            print()

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2, "sup + qry"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        # assert len(batch[0][0][0]) == 12, "data with 12 elements"

    def clone_learner(self, inference=False):
        learner = self.learner.clone()
        # if inference:
            # detach_module(learner.module, keep_requires_grad=True)
        learner.inner_freeze()
        for module in self.algorithm_config["adapt"]["modules"]:
            learner.inner_unfreeze(module)
        return learner

    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adapt_steps=5, learner=None, train=True,
              inference=False):
        """ Inner adaptation or fine-tuning. """
        if learner is None:
            learner = self.learner.clone()
            learner.inner_freeze()
            for module in self.algorithm_config["adapt"]["modules"]:
                learner.inner_unfreeze(module)
        learner.train()

        # Adapt the classifier
        sup_batch = batch[0][0][0]
        # ref_psd = sup_batch[9].unsqueeze(2) if self.reference_prosody else None
        ref_psd = (sup_batch["p_targets"].unsqueeze(2)
                   if self.reference_prosody else None)
        first_order = not train
        for step in range(adapt_steps):
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
        """
        if adapt_steps is None:
            adapt_steps = min(self.adaptation_steps, self.test_adaptation_steps)
        with torch.set_grad_enabled(train):
            learner = self.adapt(batch, adapt_steps, learner=learner, train=train)
            if not train:
                detach_module(learner.module, keep_requires_grad=True)
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
                # sup_batch[2], *qry_batch[3:],
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
                average_spk_emb=True, reference_prosody=ref_psd
            )
            valid_error = self.loss_func(qry_batch, predictions)
            if not train:
                for element in predictions:
                    try:
                        element.detach_()
                    except:
                        pass

        if synth:
            with torch.no_grad():
                if self.d_target:
                    synth_predictions = learner(
                        # sup_batch[2], *qry_batch[3:9], d_targets=qry_batch[11],
                        speaker_args=sup_batch["speaker_args"],
                        texts=qry_batch["texts"],
                        src_lens=qry_batch["src_lens"],
                        max_src_len=qry_batch["max_src_len"],
                        mels=qry_batch["mels"],
                        mel_lens=qry_batch["mel_lens"],
                        max_mel_len=qry_batch["max_mel_len"],
                        d_targets=qry_batch["d_targets"],
                        average_spk_emb=True, reference_prosody=ref_psd
                    )
                else:
                    synth_predictions = learner(
                        # sup_batch[2], *qry_batch[3:6],
                        speaker_args=sup_batch["speaker_args"],
                        texts=qry_batch["texts"],
                        src_lens=qry_batch["src_lens"],
                        max_src_len=qry_batch["max_src_len"],
                        average_spk_emb=True, reference_prosody=ref_psd
                    )
                for element in synth_predictions:
                    try:
                        element.detach_()
                    except:
                        pass
            if recon:
                return learner, valid_error, predictions, synth_predictions

            return learner, valid_error, synth_predictions

        return learner, valid_error, predictions

    def _val_step_for_fit(self, batch, batch_idx, dataloader_idx):
        """ validation_step() during Trainer.fit() """
        _, val_loss, predictions = self.meta_learn(batch, self.adaptation_steps, train=False)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=1)
        return {
            'losses': val_loss,
            'output': predictions,
            '_batch': batch[0][1][0],
        }

    def _on_val_end_for_fit(self):
        """ on_validation_end() during Trainer.fit() """
        if self.global_step > 0:
            self.print(f"Step {self.global_step} - Val losses")
            loss_dicts = []
            for i in range(len(self.trainer.datamodule.val_datasets)):
                key_map = {
                    f"Val/Total Loss/dataloader_idx_{i}": "total_loss",
                    f"Val/Mel Loss/dataloader_idx_{i}": "mel_loss",
                    f"Val/Mel-Postnet Loss/dataloader_idx_{i}": "_mel_loss",
                    f"Val/Duration Loss/dataloader_idx_{i}": "d_loss",
                    f"Val/Pitch Loss/dataloader_idx_{i}": "p_loss",
                    f"Val/Energy Loss/dataloader_idx_{i}": "e_loss",
                }
                loss_dict = {"dataloader_idx": int(i)}
                for k in key_map:
                    loss_dict[key_map[k]] = self.trainer.callback_metrics[k].item()
                loss_dicts.append(loss_dict)

            df = pd.DataFrame(loss_dicts).set_index("dataloader_idx")
            self.print(df.to_string(header=True, index=True)+'\n')

    def _val_step_for_validate(self, batch, batch_idx, dataloader_idx):
        outputs = self._test_step(batch, batch_idx, dataloader_idx)

        self.saver._on_val_batch_end_for_validate(
            self.trainer, self, outputs, batch, batch_idx, dataloader_idx)

        st = time.time()
        if "accents" in outputs["_batch"]:
            with torch.no_grad():
                recon_acc = self.eval_model(
                    outputs["step_0"]["recon"]["output"][1].transpose(1, 2))
            recon_acc_prob = F.softmax(recon_acc, dim=-1)[
                torch.arange(recon_acc.shape[0]).to(self.device),
                outputs["_batch"]["accents"]]
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
                **loss_dict
            }
            if "accents" in outputs["_batch"]:
                step_acc = self.eval_model(
                    outputs[step]["synth"]["output"][1].transpose(1, 2))
                step_acc_prob = F.softmax(step_acc, dim=-1)[
                    torch.arange(step_acc.shape[0]).to(self.device),
                    outputs["_batch"]["accents"]]
                self.log(f"{step}_acc_prob", step_acc_prob)
                self.eval_acc[step](step_acc, outputs["_batch"]["accents"])
                self.log(f"{step}_acc", self.eval_acc[step])
                data["accent_acc"] = step_acc_prob.detach().cpu().numpy().tolist()[0]
            df = pd.DataFrame([data])
            df.to_csv(self.csv_path, index=False, mode='a', header=not
                      os.path.exists(self.csv_path))
            # loss_dict = {f"{outputs['task_id']}/step_{ft_step}/{k}": v for k, v in
            #              loss_dict.items()}
            # self.log_dict(loss_dict, sync_dist=True, batch_size=1,
            #               add_dataloader_idx=False)
        self._print_mem_diff(st, "Log_dict")

        st = time.time()
        torch.cuda.empty_cache()
        self._print_mem_diff(st, "Empty cache")
        # return outputs

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
