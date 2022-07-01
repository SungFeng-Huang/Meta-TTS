import os
import pandas as pd
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile

from utils.tools import expand, plot_mel

from typing import Any
from pytorch_lightning import Trainer, LightningModule

from lightning.utils import loss2dict
# from lightning.callbacks.utils import recon_samples, synth_samples
from .base import BaseSaver, CSV_COLUMNS


class TestSaver(BaseSaver):
    """
    """

    def __init__(self,
                 preprocess_config,
                 log_dir=None,
                 result_dir=None):
        super().__init__(preprocess_config, log_dir, result_dir)

    def on_test_start(self,
                      trainer: Trainer,
                      pl_module: LightningModule):
        global_step = getattr(pl_module, 'test_global_step', pl_module.global_step)

        self.result_dir = os.path.join(trainer.log_dir, trainer.state.fn)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Result directory:", self.result_dir)

        self.test_figure_dir = os.path.join(
            self.result_dir, trainer.state.stage, "figure", f"step_{global_step}")
        self.test_audio_dir = os.path.join(
            self.result_dir, trainer.state.stage, "audio", f"step_{global_step}")
        self.test_log_dir = os.path.join(
            self.result_dir, trainer.state.stage, "csv", f"step_{global_step}")
        os.makedirs(self.test_figure_dir, exist_ok=True)
        os.makedirs(self.test_audio_dir, exist_ok=True)
        os.makedirs(self.test_log_dir, exist_ok=True)

    def on_test_batch_end(self,
                          trainer: Trainer,
                          pl_module: LightningModule,
                          all_outputs: list,
                          batch: Any,
                          batch_idx: int,
                          dataloader_idx: int):
        adaptation_steps = pl_module.adaptation_steps           # log/save period
        # test_adaptation_steps = pl_module.test_adaptation_steps # total fine-tune steps
        test_adaptation_steps = max(pl_module.saving_steps) # total fine-tune steps
        self.vocoder.to(pl_module.device)

        sup_ids = batch[0][0][0]["ids"]
        qry_ids = batch[0][1][0]["ids"]
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        _task_id = trainer.datamodule.test_SQids2Tid[SQids]

        for i, outputs in enumerate(all_outputs):
            if len(all_outputs) == 1:
                task_id = _task_id
            else:
                task_id = f"{_task_id}_{i}"

            figure_dir = os.path.join(self.test_figure_dir, task_id)
            audio_dir = os.path.join(self.test_audio_dir, task_id)
            csv_file_path = os.path.join(self.test_log_dir, f"{task_id}.csv")
            os.makedirs(figure_dir, exist_ok=True)
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

            loss_dicts = []
            _batch = outputs["_batch"]

            for ft_step in range(0, test_adaptation_steps+1, adaptation_steps):
                if ft_step == 0:
                    predictions = outputs[f"step_{ft_step}"]["recon"]["output"]
                    recon_samples(
                        _batch, predictions, self.vocoder, self.preprocess_config,
                        figure_dir, audio_dir
                    )

                if f"step_{ft_step}" in outputs:
                    if "recon" in outputs[f"step_{ft_step}"]:
                        valid_error = outputs[f"step_{ft_step}"]["recon"]["losses"]
                        loss_dicts.append({"Step": ft_step, **loss2dict(valid_error)})

                    if "synth" in outputs[f"step_{ft_step}"]:
                        predictions = outputs[f"step_{ft_step}"]["synth"]["output"]
                        synth_samples(
                            _batch, predictions, self.vocoder, self.preprocess_config,
                            figure_dir, audio_dir, f"FTstep_{ft_step}"
                        )

            df = pd.DataFrame(loss_dicts, columns=["Step"] + CSV_COLUMNS).set_index("Step")
            df.to_csv(csv_file_path, mode='a', header=True, index=True)


def recon_samples(targets, predictions, vocoder, preprocess_config, figure_dir, audio_dir):
    """Reconstruct all samples of the batch."""
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

    for i in range(len(output)):
        basename = targets["ids"][i]
        src_len = src_lens[i].item()
        mel_len = mel_lens[i].item()
        mel_target = targets["mels"][i, :mel_len].detach().transpose(0, 1)
        duration = targets["d_targets"][i, :src_len].detach().cpu().numpy()
        pitch = targets["p_targets"][i, :src_len].detach().cpu().numpy()
        energy = targets["e_targets"][i, :src_len].detach().cpu().numpy()

        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        else:
            pitch = targets["p_targets"][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        else:
            energy = targets["e_targets"][i, :mel_len].detach().cpu().numpy()

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
                (mel_target.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Ground-Truth Spectrogram"],
        )
        plt.savefig(os.path.join(figure_dir, f"{basename}.target.png"))
        plt.close()

    mel_targets = targets["mels"].transpose(1, 2)
    lengths = mel_lens * preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_targets = vocoder.infer(mel_targets, max_wav_value, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_targets, targets["ids"]):
        wavfile.write(os.path.join(audio_dir, f"{basename}.recon.wav"), sampling_rate, wav)

def synth_samples(targets, predictions, vocoder, preprocess_config, figure_dir, audio_dir, name):
    """Synthesize the first sample of the batch."""
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

    for i in range(len(output)):
        basename = targets["ids"][i]
        src_len = src_lens[i].item()
        mel_len = mel_lens[i].item()
        mel_prediction = postnet_output[i, :mel_len].detach().transpose(0, 1)
        duration = d_rounded[i, :src_len].detach().cpu().numpy()
        pitch = p_predictions[i, :src_len].detach().cpu().numpy()
        energy = e_predictions[i, :src_len].detach().cpu().numpy()

        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        else:
            pitch = targets["p_targets"][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        else:
            energy = targets["e_targets"][i, :mel_len].detach().cpu().numpy()

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
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(figure_dir, f"{basename}.{name}.synth.png"))
        plt.close()

    mel_predictions = postnet_output.transpose(1, 2)
    lengths = mel_lens * preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_predictions = vocoder.infer(mel_predictions, max_wav_value, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, targets["ids"]):
        wavfile.write(os.path.join(audio_dir, f"{basename}.{name}.synth.wav"), sampling_rate, wav)

