#!/usr/bin/env python3

import pandas as pd
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Any, Optional, Union, List
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

import torch
# from torch.nn.utils.prune import L1Unstructured, RandomUnstructured, BasePruningMethod
# from torchmetrics import Accuracy
# from pytorch_lightning.callbacks.pruning import ModelPruning, _MODULE_CONTAINERS
# from learn2learn.utils import clone_module, update_module, detach_module, clone_parameters

from lightning.algorithms.MAML import MAML, MetaSGD, AlphaMAML, AdaMAML
from lightning.systems.base_adapt.fit import BaseAdaptorFitSystem
from lightning.callbacks.saver import FitSaver
from lightning.optimizer import get_optimizer
from lightning.scheduler import get_scheduler
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.utils import loss2dict
# from .prune_accent import PruneAccentSystem
# from .utils import global_learnable_unstructured, LogitsAndMasks, setup_structured_prune
# from src.transformer.Layers import FFTBlock, MultiHeadAttention, PositionwiseFeedForward
from src.utils.tools import load_yaml
# from projects.xvec.model import XvecTDNN


torch.autograd.set_detect_anomaly(True)

class DistillSystem(BaseAdaptorFitSystem):
    """
    Inherit from BaseAdaptorSystem for `on_load_checkpoint` and `subset_step`.
    """
    def __init__(self,
                 preprocess_config: Union[dict, str],
                 model_config: Union[dict, str],
                 train_config: Union[dict, str, list],
                 algorithm_config: Union[dict, str],
                 ckpt_path: str,
                 log_dir: str = None,
                 result_dir: str = None,
                 accent_ckpt: str = "output/xvec/lightning_logs/version_88/checkpoints/epoch=199-step=164200.ckpt",
                 speaker_ckpt: str= "output/xvec/lightning_logs/version_99/checkpoints/epoch=49-step=377950.ckpt",
                 verbose: int = 2,
                 ):
        """
        Args:
            preprocess_config: Student/teacher preprocess config.
            model_config: Student model config.
            train_config: Student train config.
                Include batch size, scheduler steps, training steps, etc.
            algorithm_config: student algorithm config
            ckpt_path: teacher ckpt_path

        Attributes:
            teacher: Teacher FastSpeech2 model (load from ``ckpt_path``).
            model: Student FastSpeech2 model.
            loss_func: FastSpeech2 loss function.
            learner (MAML): MAML learner object for adaptation.
            d_target (bool): Whether use ground-truth duration for inference. Default ``False``.
            reference_prosody (bool): Whether use ground-truch prosody for prosody encoder. Only effective when algorithm_config["adapt"]["AdaLN"]["prosody"] exists. Default: ``True``.
            verbose (bool):
            inner_sched (bool): Whether use scheduler for inner-loop (fine-tune). Default ``True``.

        Expected (not implemented):
            override_kwargs: kwargs to override during inference.
            avg_train_spk_emb (bool): Initialize new speaker embeddings with the average of pretrained speaker embeddings. Default: ``True``.
            log_metrics_per_step (bool): self.log_dict(metrics) in every train/val step. Default: ``False``.
        """
        super().__init__(
            preprocess_config=preprocess_config,
            model_config=model_config,
            train_config=train_config,
            algorithm_config=algorithm_config,
            log_dir=log_dir,
            result_dir=result_dir,
        )
        # adaptation_lr = self.algorithm_config["adapt"]["task"]["lr"]
        self.learner = AdaMAML(
            self.model,
            lr=3e-5,
            adapt_transform=False,
            first_order=True,
            allow_unused=False,
            allow_nograd=True,
        )

        teacher_ckpt = torch.load(ckpt_path)
        teacher_preprocess_yaml: dict = teacher_ckpt["hyper_parameters"]["preprocess_config"]
        teacher_model_yaml: dict = teacher_ckpt["hyper_parameters"]["model_config"]
        teacher_algorithm_yaml: dict = teacher_ckpt["hyper_parameters"]["algorithm_config"]
        if isinstance(teacher_preprocess_yaml, str):
            teacher_preprocess_yaml = load_yaml(teacher_preprocess_yaml)
        if isinstance(teacher_model_yaml, str):
            teacher_model_yaml = load_yaml(teacher_model_yaml)
        if isinstance(teacher_algorithm_yaml, str):
            teacher_algorithm_yaml = load_yaml(teacher_algorithm_yaml)
        self.teacher = FastSpeech2(teacher_preprocess_yaml,
                                   teacher_model_yaml,
                                   teacher_algorithm_yaml)

        self.teacher.requires_grad_(False)
        self.teacher.eval()
        # self.model.requires_grad_(False)

        self.verbose = verbose

        # Speaker embedding weighted average
        # Case 1: average all training speaker embedding
        #   Done in `on_fit_start()`
        # No case 2

        # FT scheduler?
        # self.inner_sched = True

    def configure_callbacks(self):
        # Save figures/audios/csvs
        saver = FitSaver(self.preprocess_config, self.log_dir, self.result_dir)
        self.saver = saver

        callbacks = [saver]
        return callbacks

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = get_optimizer(self.model, self.model_config, self.train_config)

        self.scheduler = {
            "scheduler": get_scheduler(self.optimizer, self.train_config),
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            # 'monitor': self.default_monitor,
        }
        return [self.optimizer], [self.scheduler]


        # self.ft_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        # self.ft_optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=1e-1 if self.inner_sched else 3e-5)
        # self.ft_opt_init_state = self.ft_optimizer.state_dict
        # if self.inner_sched:
        #     self.ft_scheduler = get_scheduler(
        #         self.ft_optimizer,
        #         {
        #             "optimizer": {
        #                 "anneal_rate": 0.3,
        #                 "anneal_steps": [500, 700],
        #                 "warm_up_step": 200,
        #             },
        #         },
        #     )
        #     self.ft_sched_state_dict = self.ft_scheduler.state_dict()

        #     return (
        #         [self.optimizer, self.ft_optimizer],
        #         [self.scheduler, self.ft_scheduler]
        #     )
        # return (
        #     [self.optimizer, self.ft_optimizer],
        #     [self.scheduler]
        # )

        # return self.ft_optimizer

    def on_train_epoch_end(self,):
        # self.ft_optimizer.__setstate__({'state': defaultdict(dict)})
        # if self.inner_sched:
        #     # self.ft_scheduler.__setstate__({'state': defaultdict(dict)})
        #     self.ft_scheduler.load_state_dict(self.ft_sched_state_dict)
        super().on_train_epoch_end()

    def distill_step(self, batch, teacher=None, student=None):
        r"""
        Args:
            _model: Use copied model instead of self.
        """
        teacher = teacher or self.teacher
        student = student or self.model

        with torch.no_grad():
            t_preds = teacher(
                **batch,
                average_spk_emb=True,
                reference_prosody=(batch["p_targets"].unsqueeze(2)
                                if self.reference_prosody else None),
            )
            duration_target = torch.clamp(
                (torch.round(torch.exp(t_preds[4]) - 1)),
                min=0,
            )
        s_preds = student(
            **batch,
            average_spk_emb=True,
            reference_prosody=(batch["p_targets"].unsqueeze(2)
                               if self.reference_prosody else None),
        )
        gt = {
            "mels": t_preds[1], # postnet_mel_predictions
            "p_targets": t_preds[2],
            "e_targets": t_preds[3],
            "d_targets": duration_target,
        }
        loss = self.loss_func(gt, s_preds)
        return loss, t_preds, s_preds

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        """ Normal forwarding.

        Function:
            common_step(): Defined in `lightning.systems.system.System`
        """
        loss, t_preds, s_preds = self.distill_step(batch)
        # ref_psd = (batch["p_targets"].unsqueeze(2)
        #            if self.reference_prosody else None)
        # output = self(**batch, reference_prosody=ref_psd)
        # loss = self.loss_func(batch, output)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=len(batch["ids"]))
        return {
            'loss': loss[0],
            'losses': [l.detach() for l in loss],
            't_preds': [o.detach() for o in t_preds],
            'output': [o.detach() for o in s_preds],
        }