#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l
from scipy.io import wavfile
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from argparse import Namespace

from torch.utils.data import DataLoader
from pytorch_lightning.loggers.base import merge_dicts
from pytorch_lightning.utilities import rank_zero_only
from learn2learn.utils.lightning import EpisodicBatcher

from model import FastSpeech2Loss, FastSpeech2
from lightning.optimizer import get_optimizer
from lightning.scheduler import get_scheduler
from lightning.collate import get_meta_collate, get_multi_collate, get_single_collate
from lightning.sampler import GroupBatchSampler, DistributedBatchSampler
from lightning.utils import LightningMelGAN
from utils.tools import expand, plot_mel


class System(pl.LightningModule):

    default_monitor: str = "val_loss"

    def __init__(
        self,
        model=None,
        optimizer=None,
        loss_func=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        scheduler=None,
        configs=None,
        vocoder=None,
        log_dir=None,
        result_dir=None,
    ):
        super().__init__()
        preprocess_config, model_config, train_config = configs
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config

        if model is None:
            model = FastSpeech2(preprocess_config, model_config)
        if loss_func is None:
            loss_func = FastSpeech2Loss(preprocess_config, model_config)
        self.model = model
        self.loss_func = loss_func

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.vocoder = LightningMelGAN(vocoder)
        self.vocoder.freeze()
        # self.config = {} if config is None else config
        # hparams will be logged to Tensorboard as text variables.
        # summary writer doesn't support None for now, convert to strings.
        # See https://github.com/pytorch/pytorch/issues/33140
        # self.hparams = Namespace(**self.config_to_hparams(self.config))
        self.log_dir = log_dir
        self.result_dir = result_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Log directory:", self.log_dir)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_idx, train=True):
        output = self(*(batch[2:]))
        loss = self.loss_func(batch, output)
        return loss, output

    def loss2str(self, loss):
        return self.dict2str(self.loss2dict(loss))

    def loss2dict(self, loss):
        tblog_dict = {
            "Loss/total_loss"       : loss[0].item(),
            "Loss/mel_loss"         : loss[1].item(),
            "Loss/mel_postnet_loss" : loss[2].item(),
            "Loss/pitch_loss"       : loss[3].item(),
            "Loss/energy_loss"      : loss[4].item(),
            "Loss/duration_loss"    : loss[5].item(),
        }
        return tblog_dict
    
    def dict2loss(self, tblog_dict):
        loss = (
            tblog_dict["Loss/total_loss"],
            tblog_dict["Loss/mel_loss"],
            tblog_dict["Loss/mel_postnet_loss"],
            tblog_dict["Loss/pitch_loss"],
            tblog_dict["Loss/energy_loss"],
            tblog_dict["Loss/duration_loss"],
        )
        return loss

    def dict2str(self, tblog_dict):
        def convert_key(key):
            new_key = ' '.join([e.title() for e in key.split('/')[-1].split('_')])
            return new_key
        message = ", ".join([f"{convert_key(k)}: {v:.4f}" for k, v in tblog_dict.items()])
        return message

    def training_step(self, batch, batch_idx):
        loss, output = self.common_step(batch, batch_idx, train=True)

        # Synthesis one sample and log to CometLogger
        if (self.global_step+1) % self.train_config["step"]["synth_step"] == 0 and self.local_rank == 0:
            fig, wav_reconstruction, wav_prediction, basename = self.synth_one_sample(batch, output)
            step = self.global_step+1
            self.log_figure(
                f"Training/step_{step}_{basename}", fig
            )
            self.log_audio(
                f"Training/step_{step}_{basename}_reconstructed", wav_reconstruction
            )
            self.log_audio(
                f"Training/step_{step}_{basename}_synthesized", wav_prediction
            )

        # Log message to log.txt and print to stdout
        if (self.global_step+1) % self.trainer.log_every_n_steps == 0:
            message = f"Step {self.global_step+1}/{self.trainer.max_steps}, "
            message += self.loss2str(loss)
            self.log_text(message)

        # Log metrics to CometLogger
        comet_log_dict = {f"train_{k}":v for k,v in self.loss2dict(loss).items()}
        self.log_dict(comet_log_dict, sync_dist=True)
        return loss[0]

    def validation_step(self, batch, batch_idx):
        loss, output = self.common_step(batch, batch_idx, train=False)
        tblog_dict = self.loss2dict(loss)

        if batch_idx == 0:
            fig, wav_reconstruction, wav_prediction, basename = self.synth_one_sample(batch, output)
            step = self.global_step+1
            self.log_figure(f"Validation/step_{step}_{basename}", fig)
            self.log_audio(f"Validation/step_{step}_{basename}_reconstructed", wav_reconstruction)
            self.log_audio(f"Validation/step_{step}_{basename}_synthesized", wav_prediction)

        total_loss = loss[0]
        self.log('val_loss', total_loss)
        return tblog_dict

    def validation_epoch_end(self, val_dicts=None):
        """Log hp_metric to tensorboard for hparams selection."""
        if self.global_step > 0:
            tblog_dict = merge_dicts(val_dicts)
            loss = self.dict2loss(tblog_dict)

            message = f"Validation Step {self.global_step+1}, "
            tqdm.write(message + self.loss2str(loss))

            self.logger[1].log_metrics(tblog_dict, self.global_step+1)

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.optimizer is None:
            self.optimizer = get_optimizer(self.model, self.model_config, self.train_config)

        if self.scheduler is None:
            self.scheduler = {
                "scheduler": get_scheduler(self.optimizer, self.train_config),
                'interval': 'step',
                'frequency': 1,
                'monitor': self.default_monitor,
            }
        else:
            if not isinstance(self.scheduler, dict):
                self.scheduler = {
                    "scheduler": self.scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    "monitor": self.default_monitor
                }
            else:
                self.scheduler.setdefault("monitor", self.default_monitor)
                self.scheduler.setdefault("frequency", 1)
                assert self.scheduler["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"

        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        """Training dataloader"""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=get_single_collate(False),
            num_workers=8,
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            collate_fn=get_single_collate(False),
            num_workers=8,
        )
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["preprocess_config"] = self.preprocess_config
        checkpoint["train_config"] = self.train_config
        checkpoint["model_config"] = self.model_config
        checkpoint["log_dir"] = self.log_dir
        return checkpoint

    # def on_load_checkpoint(self, checkpoint):

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.preprocess_config = checkpoint["preprocess_config"]
        self.train_config = checkpoint["train_config"]
        self.model_config = checkpoint["model_config"]
        self.log_dir = checkpoint["log_dir"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    try:
                        self.print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    except:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                try:
                    self.print(f"Dropping parameter {k}")
                except:
                    print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    @rank_zero_only
    def synth_one_sample(self, targets, predictions):
        """Synthesize the first sample of the batch."""
        basename = targets[0][0]
        src_len = predictions[8][0].item()
        mel_len = predictions[9][0].item()
        mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
        mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
        duration = targets[11][0, :src_len].detach().cpu().numpy()
        if self.preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = targets[9][0, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = targets[9][0, :mel_len].detach().cpu().numpy()
        if self.preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = targets[10][0, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = targets[10][0, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(self.preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
                (mel_target.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
        )

        if self.vocoder.mel2wav is not None:
            max_wav_value = self.preprocess_config["preprocessing"]["audio"]["max_wav_value"]

            wav_reconstruction = self.vocoder.infer(mel_target.unsqueeze(0), max_wav_value)[0]
            wav_prediction = self.vocoder.infer(mel_prediction.unsqueeze(0), max_wav_value)[0]
        else:
            wav_reconstruction = wav_prediction = None

        return fig, wav_reconstruction, wav_prediction, basename

    def log_audio(self, tag, audio, metadata=None):
        train = self.trainer.training
        step = self.global_step+1
        sample_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        stage, basename = tag.split('/', 1)
        os.makedirs(os.path.join(self.log_dir, "audio", stage), exist_ok=True)
        wavfile.write(os.path.join(self.log_dir, "audio", f"{tag}.wav"), sample_rate, audio)
        if hasattr(self, 'logger'):
            if isinstance(self.logger[0], pl.loggers.CometLogger):
                basename = basename.split('_')
                idx = '_'.join(basename[2:-1])
                if metadata is None:
                    metadata = {'stage': stage}
                else:
                    metadata = metadata.copy()
                audio_type = basename[-1]
                metadata.update({'type': audio_type, 'id': idx})
                file_name = f"{idx}_{audio_type}.wav"

                self.logger[0].experiment.log_audio(
                    audio_data=audio / max(abs(audio)),
                    sample_rate=sample_rate,
                    file_name=file_name,
                    step=step,
                    metadata=metadata,
                )
            elif isinstance(self.logger[int(not train)], pl.loggers.TensorBoardLogger):
                self.logger[int(not train)].experiment.add_audio(
                    tag=tag,
                    snd_tensor=audio / max(abs(audio)),
                    global_step=step,
                    sample_rate=sample_rate,
                )
            else:
                self.print("Failed to log audio: not finding correct logger type")

    def log_figure(self, tag, figure):
        train = self.trainer.training
        step = self.global_step+1
        stage, basename = tag.split('/', 1)
        # os.makedirs(os.path.join(self.log_dir, "figure", stage), exist_ok=True)
        # wavfile.write(os.path.join(self.log_dir, "figure", f"{tag}.png"), sample_rate, audio)
        if isinstance(self.logger[0], pl.loggers.CometLogger):
            self.logger[0].experiment.log_figure(
                figure_name=tag,
                figure=figure,
                step=step,
            )
        elif isinstance(self.logger[int(not train)], pl.loggers.TensorBoardLogger):
            self.logger[int(not train)].experiment.add_figure(
                tag=tag,
                figure=figure,
                global_step=step,
            )
        else:
            self.print("Failed to log figure: not finding correct logger type")

    def log_text(self, text):
        self.print(text)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            f.write(text + '\n')


class FewShotSystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _few_shot_task_dataset(self, _dataset, ways, shots, queries, task_per_speaker=-1, epoch_length=-1):
        # Make meta-dataset, to apply 1-way-5-shots tasks
        id2lb = {k:v for k,v in enumerate(_dataset.speaker)}
        meta_dataset = l2l.data.MetaDataset(_dataset, indices_to_labels=id2lb)

        if task_per_speaker > 0:
            # 1-way-5-shots tasks for each speaker
            tasks = []
            for label, indices in meta_dataset.labels_to_indices.items():
                if len(indices) >= shots+queries:
                    transforms = [
                        l2l.data.transforms.FusedNWaysKShots(
                            meta_dataset,
                            n=ways,
                            k=shots+queries,
                            replacement=False,
                            filter_labels=[label]
                        ),
                        l2l.data.transforms.LoadData(meta_dataset),
                    ]
                    _tasks = l2l.data.TaskDataset(
                        meta_dataset,
                        task_transforms=transforms,
                        task_collate=get_meta_collate(shots, queries, False),
                        num_tasks=task_per_speaker,
                    )
                    tasks.append(_tasks)
            tasks = torch.utils.data.ConcatDataset(tasks)
        else:
            # 1-way-5-shots transforms
            transforms = [
                l2l.data.transforms.FusedNWaysKShots(
                    meta_dataset,
                    n=ways,
                    k=shots+queries,
                    replacement=True
                ),
                l2l.data.transforms.LoadData(meta_dataset),
            ]

            # 1-way-5-shot task dataset
            tasks = l2l.data.TaskDataset(
                meta_dataset,
                task_transforms=transforms,
                task_collate=get_meta_collate(shots, queries, False),
            )

            if epoch_length > 0:
                # Epochify task dataset, for periodic validation
                tasks = EpisodicBatcher(
                    tasks, epoch_length=epoch_length,
                )
        return tasks

    def prefetch_tasks(self, tasks, tag='val', log_dir=''):
        os.makedirs(os.path.join(log_dir, f"{tag}_csv"), exist_ok=True)

        SQids = []
        SQids2Tid = {}
        for i, task in enumerate(tasks):
            sup_ids = task[0][0][0]
            qry_ids = task[1][0][0]
            SQids.append({'sup_id': sup_ids, 'qry_id': qry_ids})

            SQid = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
            SQids2Tid[SQid] = f"{tag}_{i:03d}"

        with open(os.path.join(log_dir, f"{tag}_SQids.json"), 'w') as f:
            json.dump(SQids, f, indent=4)

        return SQids2Tid

    def _on_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2, "sup + qry"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 12, "data with 12 elements"

    def log_csv(self, losses, log_dir='', file_name='', step=0):

        os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(os.path.join(log_dir, file_name)):
            with open(os.path.join(log_dir, file_name), 'w') as f:
                f.write(
                    "step, total_loss, mel_loss, mel_postnet_loss, pitch_loss, energy_loss, duration_loss\n"
                )

        with open(os.path.join(log_dir, file_name), 'a') as f:
            f.write(f"{step}")
            for loss in losses:
                f.write(f", {loss.item()}")
            f.write("\n")

    def recon_samples(self, targets, predictions, tag='Testing', log_dir=''):
        """Synthesize the first sample of the batch."""
        for i in range(len(predictions[0])):
            basename    = targets[0][i]
            src_len     = predictions[8][i].item()
            mel_len     = predictions[9][i].item()
            mel_target  = targets[6][i, :mel_len].detach().transpose(0, 1)
            duration    = targets[11][i, :src_len].detach().cpu().numpy()
            pitch       = targets[9][i, :src_len].detach().cpu().numpy()
            energy      = targets[10][i, :src_len].detach().cpu().numpy()
            if self.preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
                pitch = expand(pitch, duration)
            if self.preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
                energy = expand(energy, duration)

            with open(
                os.path.join(self.preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                stats = stats["pitch"] + stats["energy"][:2]

            fig = plot_mel(
                [
                    (mel_target.cpu().numpy(), pitch, energy),
                ],
                stats,
                ["Ground-Truth Spectrogram"],
            )
            os.makedirs(os.path.join(log_dir, "figure", f"{tag}"), exist_ok=True)
            plt.savefig(os.path.join(log_dir, "figure", f"{tag}/{basename}.target.png"))
            plt.close()

        mel_targets = targets[6].transpose(1, 2)
        lengths = predictions[9] * self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        max_wav_value = self.preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        wav_targets = self.vocoder.infer(mel_targets, max_wav_value, lengths=lengths)

        sampling_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        os.makedirs(os.path.join(log_dir, "audio", f"{tag}"), exist_ok=True)
        for wav, basename in zip(wav_targets, targets[0]):
            wavfile.write(os.path.join(log_dir, "audio", f"{tag}/{basename}.recon.wav"), sampling_rate, wav)

    def synth_samples(self, targets, predictions, tag='Testing', name='', log_dir=''):
        """Synthesize the first sample of the batch."""
        for i in range(len(predictions[0])):
            basename        = targets[0][i]
            src_len         = predictions[8][i].item()
            mel_len         = predictions[9][i].item()
            mel_prediction  = predictions[1][i, :mel_len].detach().transpose(0, 1)
            duration        = predictions[5][i, :src_len].detach().cpu().numpy()
            pitch           = predictions[2][i, :src_len].detach().cpu().numpy()
            energy          = predictions[3][i, :src_len].detach().cpu().numpy()
            if self.preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
                pitch = expand(pitch, duration)
            if self.preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
                energy = expand(energy, duration)

            with open(
                os.path.join(self.preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                stats = stats["pitch"] + stats["energy"][:2]

            fig = plot_mel(
                [
                    (mel_prediction.cpu().numpy(), pitch, energy),
                ],
                stats,
                ["Synthetized Spectrogram"],
            )
            os.makedirs(os.path.join(log_dir, "figure", f"{tag}"), exist_ok=True)
            plt.savefig(os.path.join(log_dir, "figure", f"{tag}/{basename}.{name}.synth.png"))
            plt.close()

        mel_predictions = predictions[1].transpose(1, 2)
        lengths = predictions[9] * self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        max_wav_value = self.preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        wav_predictions = self.vocoder.infer(mel_predictions, max_wav_value, lengths=lengths)

        sampling_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        os.makedirs(os.path.join(log_dir, "audio", f"{tag}"), exist_ok=True)
        for wav, basename in zip(wav_predictions, targets[0]):
            wavfile.write(os.path.join(log_dir, "audio", f"{tag}/{basename}.{name}.synth.wav"), sampling_rate, wav)


