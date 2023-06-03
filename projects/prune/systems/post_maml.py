import os
import pandas as pd
from pytorch_lightning.callbacks.base import Callback
import torch
import torch.distributed as torch_distrib
import pytorch_lightning as pl
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pytorch_lightning.utilities.cli import instantiate_class

from lightning.systems.base_adapt import BaseAdaptorSystem, BaselineFitSystem
from lightning.algorithms.MAML import AdamWMAML, MAML
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.utils import loss2dict
from projects.prune.ckpt_utils import prune_model
from src.utils.tools import load_yaml
from lightning.scheduler import get_scheduler
from src.utils.tools import load_yaml

import gc


class PrunedMetaSystem(BaseAdaptorSystem):
    def __init__(
        self,
        pruned_ckpt_path: str,
        algorithm_config: Union[dict, str],
        log_dir: str = None,
        result_dir: str = None,
    ):
        pl.LightningModule.__init__(self)
        if isinstance(algorithm_config, str):
            algorithm_config = load_yaml(algorithm_config)
        self.algorithm_config = algorithm_config
        self.save_hyperparameters()

        # ckpt = torch.load(ckpt_path)
        # preprocess_yaml: str = ckpt["hyper_parameters"]["preprocess_config"]
        # model_yaml: str = ckpt["hyper_parameters"]["model_config"]
        # model = FastSpeech2(
        #     load_yaml(preprocess_yaml),
        #     load_yaml(model_yaml),
        #     load_yaml(ckpt["hyper_parameters"]["algorithm_config"])
        # ).cuda()
        # self.model = prune_model(model, ckpt).eval()

        self.model: FastSpeech2 = torch.load(pruned_ckpt_path)
        self.loss_func = FastSpeech2Loss(
            preprocess_config={"preprocessing": {
                "energy": {"feature": self.model.variance_adaptor.energy_feature_level},
                "pitch": {"feature": self.model.variance_adaptor.pitch_feature_level},
            }},
            model_config=None,
        )

        self.log_dir = log_dir
        self.result_dir = result_dir

        # All of the settings below are for few-shot validation
        self.learner = instantiate_class(self.model, self.algorithm_config["adapt"]["learner_config"])
        if isinstance(self.learner, AdamWMAML):
            self.learner.compute_update.transforms_modules.requires_grad_(False)
        self.adaptation_steps: int = self.algorithm_config["adapt"]["train"].get("steps", 5)
        # self.adaptation_steps: int = self.algorithm_config["adapt"]["test"].get("steps", 5)

        self.d_target = False
        self.reference_prosody = True

        self.automatic_optimization = False

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return super().configure_callbacks()
    
    def configure_optimizers(self):
        """
        Optimizers for self.model, self.learner.compute_update, (self.logits_and_masks).
        Returns:
            A list of optimizers.
        """
        # TODO
        tts_opt = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        tts_scheduler = get_scheduler(
            tts_opt,
            {
                "optimizer": {
                    "anneal_rate": 0.3,
                    "anneal_steps": [1000, 2000, 3000],
                    "warm_up_step": 600,
                },
            },
        )
        if isinstance(self.learner, MAML):
            return [tts_opt], [tts_scheduler]
        else:
            maml_opt = torch.optim.Adam(self.learner.compute_update.parameters(), lr=3e-4)
            maml_scheduler = get_scheduler(
                maml_opt,
                {
                    "optimizer": {
                        "anneal_rate": 0.3,
                        "anneal_steps": [1000, 2000, 3000],
                        "warm_up_step": 600,
                    },
                },
            )
            return [tts_opt, maml_opt], [tts_scheduler, maml_scheduler]

    def setup(self, stage=None):
        self.meta_batch_size = self.algorithm_config["adapt"]["train"].get("meta_batch_size", 1)
        self.accumulate_grad_batches = self.meta_batch_size // self.trainer.world_size
        self.avg_spk_emb()
        return super().setup(stage)

    def forward(self, *args, **kwargs): return super().forward(*args, **kwargs)

    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def meta_adapt(self, batch, adapt_steps=5, learner=None, train=True, multi_step_loss=False):
        """ Inner adaptation or fine-tuning. """
        if learner is None:
            learner = self.clone_learner(inference=not train)

        # Adapt the classifier
        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]

        def sup_adapt(sup_batch):
            first_order = not train
            ref_psd = (sup_batch["p_targets"].unsqueeze(2)
                       if self.reference_prosody else None)
            learner.train()
            preds = learner(**sup_batch, reference_prosody=ref_psd)
            train_error = self.loss_func(sup_batch, preds)
            learner.adapt_(
                train_error[0],
                first_order=first_order, allow_unused=True, allow_nograd=True
            )

        def qry_backward(sup_batch, qry_batch, retain_graph=True):
            learner.eval()
            qry_ref_psd = (qry_batch["p_targets"].unsqueeze(2)
                           if self.reference_prosody else None)
            qry_kw = ["texts", "src_lens", "max_src_len", "mels", "mel_lens", "max_mel_len", "p_targets", "e_targets", "d_targets"]
            qry_kwargs = {
                k: qry_batch[k] if k in qry_kw and k in qry_batch else sup_batch[k] for k in sup_batch
            }
            qry_kwargs["reference_prosody"] = qry_ref_psd
            preds = learner(**qry_kwargs, average_spk_emb=True)
            qry_loss = self.loss_func(qry_batch, preds)
            if train:
                self.manual_backward(qry_loss[0], retain_graph=retain_graph)
            return qry_loss, preds

        for i in range(adapt_steps):
            sup_adapt(sup_batch)
            if multi_step_loss and i != adapt_steps - 1:
                qry_backward(sup_batch, qry_batch, retain_graph=train)
            gc.collect()
            torch.cuda.empty_cache()
        qry_loss, preds = qry_backward(sup_batch, qry_batch, retain_graph=False)
        return learner, qry_loss, preds

    def avg_spk_emb(self):
        with torch.no_grad():
            self.model.speaker_emb.model.weight[2311:] = \
                self.model.speaker_emb.model.weight[:2311].mean(dim=0)

    # def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # return super().on_after_batch_transfer(batch, dataloader_idx)

    def on_train_batch_start(self, batch, batch_idx):
        """Log global_step to progress bar instead of GlobalProgressBar
        callback.
        """
        # self.log("global_step", self.global_step, prog_bar=True)
        try:
            self.trainer.progress_bar_callback.train_progress_bar.set_description(f"Epoch: {self.current_epoch}, {self.global_step}/{self.trainer.progress_bar_callback.total_train_batches}")
        except:
            self.trainer.progress_bar_callback.main_progress_bar.set_description(f"Epoch: {self.current_epoch}, {self.global_step}/{self.trainer.progress_bar_callback.total_train_batches}")

    def training_step(self, batch, batch_idx):
        """ Normal forwarding.

        Function:
            common_step(): Defined in `lightning.systems.system.System`
        """
        if isinstance(self.learner, MAML):
            tts_opt = self.optimizers()
            tts_sch = self.lr_schedulers()
        else:
            tts_opt, maml_opt = self.optimizers()
            tts_sch, maml_sch = self.lr_schedulers()

        # DDP sync grads already done by self.manual_backward
        _, train_loss, predictions = self.meta_adapt(batch, adapt_steps=self.adaptation_steps, train=True, multi_step_loss=False)

        # if (self.global_step+1) % self.accumulate_grad_batches == 0:
        if (batch_idx+1) % self.accumulate_grad_batches == 0:
            # Update
            self.clip_gradients(tts_opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
            tts_opt.step()
            tts_sch.step()
            if not isinstance(self.learner, MAML):
                self.clip_gradients(maml_opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
                maml_opt.step()
                maml_sch.step()
            self.learner.zero_grad()
            self.avg_spk_emb()

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=1, prog_bar=True)
        return {
            'loss': train_loss[0],
            'losses': [loss.detach() for loss in train_loss],
            'output': [p.detach() for p in predictions],
            '_batch': batch[0][1][0],   # qry_batch
        }
        
    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        metrics = self.trainer.progress_bar_callback.get_metrics(
            self.trainer, self
        )
        key_map = {
            "Train/Total Loss": "tot_L",
            "Train/Mel Loss": "ms_L",
            "Train/Mel-Postnet Loss": "_ms_L",
            "Train/Duration Loss": "d_L",
            "Train/Pitch Loss": "p_L",
            "Train/Energy Loss": "e_L",
        }
        # print(metrics.keys())
        for k in key_map:
            metrics[key_map[k]] = metrics[k]
            metrics.pop(k)
        try:
            self.trainer.progress_bar_callback.main_progress_bar.set_postfix(metrics)
        except:
            self.trainer.progress_bar_callback.train_progress_bar.set_postfix(metrics)

    def on_train_epoch_end(self):
        """Print logged metrics during on_train_epoch_end()."""
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

        df = pd.DataFrame([loss_dict]).set_index("step")
        if (self.current_epoch ) % self.trainer.check_val_every_n_epoch == 0:
            self.print(df.to_string(header=True, index=True))
        else:
            self.print(df.to_string(header=True, index=True).split('\n')[-1])

    def on_validation_start(self,):
        self.mem_stats = torch.cuda.memory_stats(self.device)
        self.csv_path = os.path.join(self.log_dir, "csv",
                                     f"validate-{self.global_step}.csv")
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ LightningModule interface.
        Operates on a single batch of data from the validation set.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
        _, val_loss, predictions = self.meta_adapt(batch, self.adaptation_steps, train=False)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=1)
        return {
            'losses': val_loss,
            'output': predictions,
            # '_batch': batch[0][1][0],
        }

    def on_validation_end(self):
        """ LightningModule interface.
        Called at the end of validation.

        self.trainer.state.fn: ("fit", "validate", "test", "predict", "tune")
        """
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

