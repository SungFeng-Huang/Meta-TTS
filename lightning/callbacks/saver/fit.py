import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import os
import json
import numpy as np

from typing import Any, Union, List, Tuple, Mapping
from pytorch_lightning import Trainer, LightningModule

from utils.tools import expand, plot_mel
from lightning.utils import loss2dict
# from lightning.callbacks.utils import synth_one_sample_with_target
from .base import BaseSaver


class FitSaver(BaseSaver):
    """
    """

    def __init__(self, preprocess_config, log_dir=None, result_dir=None):
        super().__init__(preprocess_config, log_dir, result_dir)

    def on_fit_start(self,
                     trainer: Trainer,
                     pl_module: LightningModule):
        self.log_dir = os.path.join(trainer.log_dir, trainer.state.fn)
        os.makedirs(self.log_dir, exist_ok=True)
        print("Log directory:", self.log_dir)

    def on_train_batch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           outputs: Mapping,
                           batch: Any,
                           batch_idx: int):
        output = outputs['output']
        logger = pl_module.logger
        self.vocoder.to(pl_module.device)
        _batch = batch
        metadata = {}
        if isinstance(batch, tuple) or isinstance(batch, list): # qry_batch
            _batch = batch[0][1][0]
            metadata = {'sup_ids': batch[0][0][0]["ids"]}

        # Synthesis one sample and log to CometLogger
        if (pl_module.global_step % pl_module.train_config["step"]["synth_step"] == 0
                and pl_module.local_rank == 0):
            fig, wav_reconstruction, wav_prediction, basename = synth_one_sample_with_target(
                _batch, output, self.vocoder, self.preprocess_config
            )
            figure_name = f"{trainer.state.stage}/{basename}"
            self.log_figure(logger, figure_name, fig, pl_module.global_step)
            # self.log_audio(logger, "Training", pl_module.global_step, basename, "reconstructed", wav_reconstruction, metadata)
            # self.log_audio(logger, "Training", pl_module.global_step, basename, "synthesized", wav_prediction, metadata)
            self.log_audio(logger, trainer.state.stage, pl_module.global_step,
                           basename, "reconstructed", wav_reconstruction, metadata)
            self.log_audio(logger, trainer.state.stage, pl_module.global_step,
                           basename, "synthesized", wav_prediction, metadata)
            plt.close(fig)

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Mapping,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int):
        loss = outputs['losses']
        output = outputs['output']
        logger = pl_module.logger
        self.vocoder.to(pl_module.device)
        if isinstance(batch, tuple) or isinstance(batch, list): # qry_batch
            _batch = batch[0][1][0]
        else:   # batch, probably never used
            _batch = batch

        # Log loss for each sample to csv files
        SQids = '.'.join(['-'.join(batch[0][0][0]["ids"]),
                          '-'.join(batch[0][1][0]["ids"])])
        task_id = trainer.datamodule.val_SQids2Tid[SQids]
        # self.log_csv("Validation", pl_module.global_step, task_id, loss2dict(loss))
        self.log_csv(trainer.state.stage, pl_module.global_step, task_id, loss2dict(loss))

        # Log figure/audio to logger + save audio
        if batch_idx == 0 and pl_module.local_rank == 0:
            metadata = {'dataloader_idx': dataloader_idx,
                        'sup_ids': batch[0][0][0]["ids"]}
            fig, wav_reconstruction, wav_prediction, basename = synth_one_sample_with_target(
                _batch, output, self.vocoder, self.preprocess_config
            )
            figure_name = f"{trainer.state.stage}/{basename}"
            self.log_figure(logger, figure_name, fig, pl_module.global_step)
            # self.log_audio(logger, "Validation", pl_module.global_step, basename, "reconstructed", wav_reconstruction, metadata)
            # self.log_audio(logger, "Validation", pl_module.global_step, basename, "synthesized", wav_prediction, metadata)
            self.log_audio(logger, trainer.state.stage, pl_module.global_step,
                           basename, "reconstructed", wav_reconstruction, metadata)
            self.log_audio(logger, trainer.state.stage, pl_module.global_step,
                           basename, "synthesized", wav_prediction, metadata)
            plt.close(fig)


def synth_one_sample_with_target(targets, predictions, vocoder, preprocess_config):
    """Synthesize the first sample of the batch given target pitch/duration/energy."""
    (
        output,
        postnet_output,
        p_predictions,
        e_predictions,
        log_d_predictions,
        d_rounded,
        src_masks,
        mel_masks,
        src_lens,
        mel_lens,
    ) = predictions

    basename = targets["ids"][0]
    src_len = src_lens[0].item()
    mel_len = mel_lens[0].item()
    mel_target = targets["mels"][0, :mel_len].detach().transpose(0, 1)
    duration = targets["d_targets"][0, :src_len].detach().cpu().numpy()
    pitch = targets["p_targets"][0, :src_len].detach().cpu().numpy()
    energy = targets["e_targets"][0, :src_len].detach().cpu().numpy()
    mel_prediction  = postnet_output[0, :mel_len].detach().transpose(0, 1)

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = expand(pitch, duration)
    else:
        pitch = targets["p_targets"][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = expand(energy, duration)
    else:
        energy = targets["e_targets"][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        # stats = stats["pitch"] + stats["energy"][:2]
    if preprocess_config["preprocessing"]["pitch"]["log"]:
        pitch = np.exp(pitch)
    elif preprocess_config["preprocessing"]["pitch"]["normalization"]:
        pitch = pitch * stats["pitch"]["std"] + stats["pitch"]["mean"]
    if preprocess_config["preprocessing"]["energy"]["log"]:
        energy = np.exp(energy)
    elif preprocess_config["preprocessing"]["energy"]["normalization"]:
        energy = energy * stats["energy"]["std"] + stats["energy"]["mean"]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder.mel2wav is not None:
        max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]

        wav_reconstruction = vocoder.infer(mel_target.unsqueeze(0), max_wav_value)[0]
        wav_prediction = vocoder.infer(mel_prediction.unsqueeze(0), max_wav_value)[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename
