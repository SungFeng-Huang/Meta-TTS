import os
import re
import pandas as pd
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile

from typing import Any, Mapping, Literal
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.pruning import ModelPruning, _MODULE_CONTAINERS

from src.utils.tools import expand, plot_mel
from lightning.utils import loss2dict
from .base import BaseSaver


class PruneAccentSaver(BaseSaver):
    """
    """

    def __init__(self,
                 preprocess_config,
                 log_dir=None,
                 result_dir=None):
        super().__init__(preprocess_config, log_dir, result_dir)

    def on_fit_start(self,
                     trainer: Trainer,
                     pl_module: LightningModule):
        self.log_dir = os.path.join(trainer.log_dir, trainer.state.fn)
        os.makedirs(self.log_dir, exist_ok=True)
        print("Log directory:", self.log_dir)
        self.csv_dir_dict = {
            stage: os.path.join(self.log_dir, stage, "csv")
            for stage in {
                "train", "sanity_check", "validate", "test", "predict", "tune"
            }
        }
        self.audio_dir_dict = {
            stage: os.path.join(self.log_dir, stage, "audio")
            for stage in {
                "train", "sanity_check", "validate", "test", "predict", "tune"
            }
        }
        self.batch_idx = -1

        import logging

        logger = logging.getLogger("pytorch_lightning.callbacks.pruning")
        pruning_log_file = os.path.join(self.log_dir, "pruning.log")
        logger.addHandler(logging.FileHandler(pruning_log_file))

    def save_wavs(self,
                  type: str,
                  trainer: Trainer,
                  pl_module: LightningModule,
                  _batch: Any,
                  _preds: Any,
                  batch_idx: int,
                  check_recon: bool = True) -> None:

        stage = trainer.state.stage
        epoch = trainer.current_epoch
        try:
            global_step = pl_module.global_step
        except:
            global_step = 0
        logger = trainer.logger

                  # type: Literal["support", "query", "validate"],
        # save_recon = {
        #     "support": global_step == 0,
        #     "query": global_step == 0,
        #     "validate": global_step == 0,
        #     "mask": epoch == 0,
        # }[type]

        _audio_dir = os.path.join(self.audio_dir_dict[stage], type)

        # if save_recon:
        #     # reconstruct from target mel-spectrogram
        #     _wav_recons, _ids, sampling_rate = recon_wavs(
        #         _batch, None, self.vocoder, self.preprocess_config, _audio_dir,
        #         postfix=".recon", save=True)
        #     # for wav, _id in zip(_wav_recons, _ids):
        #     #     file_name=f"{type}-{_id}/recon"
        #     #     self._log_audio(
        #     #         logger, file_name=file_name, audio=wav, step=global_step,
        #     #         sample_rate=sampling_rate)

        _postfix = f"-epoch={epoch}-batch_idx={batch_idx}"
        _wav_preds, _ids, sampling_rate = synth_wavs(
            _batch, _preds, self.vocoder, self.preprocess_config, _audio_dir,
            postfix=_postfix, save=True, check_recon=check_recon)
        # for wav, _id in zip(_wav_preds, _ids):
        #     file_name=f"{type}-{_id}/epoch={epoch}/batch_idx={batch_idx}"
        #     self._log_audio(
        #         logger, file_name=file_name, audio=wav, step=global_step,
        #         sample_rate=sampling_rate)

    def save_sup_wavs(self,
                      trainer: Trainer,
                      pl_module: LightningModule,
                      sup_batch: Any,
                      sup_preds: Any,
                      batch_idx: int) -> None:
        return self.save_wavs(
            "support", trainer, pl_module, sup_batch, sup_preds, batch_idx)

    def save_qry_wavs(self,
                      trainer: Trainer,
                      pl_module: LightningModule,
                      qry_batch: Any,
                      qry_preds: Any,
                      batch_idx: int) -> None:
        return self.save_wavs(
            "query", trainer, pl_module, qry_batch, qry_preds, batch_idx)

    def on_train_batch_start(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           batch: Any,
                           batch_idx: int):
        if batch_idx == 0:
            print(trainer.state.stage)
            print(pl_module.model.mel_linear.weight)
            print((pl_module.model.mel_linear.weight == 0).sum().item())

    def on_train_batch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           outputs: Mapping,
                           batch: Any,
                           batch_idx: int):
        if outputs is None or len(outputs) == 0:
            return

        self.vocoder.to(pl_module.device)
        self.vocoder.eval()

        sup_batch, qry_batch = batch[0]["sup"], batch[0]["qry"]
        sup_preds = outputs["sup_preds"]
        qry_preds = outputs["qry_preds"]

        if (batch_idx == 0
                or trainer.is_last_batch
                or pl_module._check_epoch_done(batch_idx+1)):
            # sup_preds
            self.save_sup_wavs(trainer, pl_module, sup_batch, sup_preds, batch_idx)
            # qry_preds
            self.save_qry_wavs(trainer, pl_module, qry_batch, qry_preds, batch_idx)

            if (trainer.is_last_batch
                    or pl_module._check_epoch_done(batch_idx+1)):
                self.batch_idx = batch_idx

        # save metrics to csv file
        epoch = trainer.current_epoch
        stage = trainer.state.stage

        qry_loss_dict = loss2dict(outputs["qry_losses"])
        qry_loss_dict.update({
            k: outputs[k] for k in {
                "qry_accent_prob",
                "qry_accent_acc",
                "qry_speaker_prob",
                "qry_speaker_acc",
            }
        })
        df = pd.DataFrame([{
            "batch_idx": batch_idx,
            **qry_loss_dict,
        }])
        _csv_dir = os.path.join(self.csv_dir_dict[stage], "query")
        csv_file_path = os.path.join(_csv_dir, f"epoch={epoch}.csv")
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df.to_csv(csv_file_path, mode='a', index=False,
                  header=not os.path.exists(csv_file_path))

        sup_loss_dict = loss2dict(outputs["sup_losses"])
        sup_loss_dict.update({
            k: outputs[k] for k in {
                "sup_accent_prob",
                "sup_accent_acc",
                "sup_speaker_prob",
                "sup_speaker_acc",
            }
        })
        df = pd.DataFrame([{
            "batch_idx": batch_idx,
            **sup_loss_dict,
        }])
        _csv_dir = os.path.join(self.csv_dir_dict[stage], "support")
        csv_file_path = os.path.join(_csv_dir, f"epoch={epoch}.csv")
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df.to_csv(csv_file_path, mode='a', index=False,
                  header=not os.path.exists(csv_file_path))

    def save_val_wavs(self,
                      trainer: Trainer,
                      pl_module: LightningModule,
                      val_batch: Any,
                      val_preds: Any,
                      epoch_start: bool) -> None:
        batch_idx = 0 if epoch_start else self.batch_idx
        return self.save_wavs(
            "validate", trainer, pl_module, val_batch, val_preds, batch_idx)

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Mapping,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int):
        self.vocoder.to(pl_module.device)
        self.vocoder.eval()

        epoch = trainer.current_epoch
        stage = trainer.state.stage

        epoch_start = outputs["epoch_start"]
        tag = "start" if epoch_start else "end"
        preds = outputs["preds"]
        self.save_val_wavs(trainer, pl_module, batch, preds, epoch_start)

        if batch_idx == 0:
            print(trainer.state.stage)
            print(pl_module.model.mel_linear.weight)
            print((pl_module.model.mel_linear.weight == 0).sum().item())

        # csv
        val_ids = batch["ids"]
        accent_prob = outputs["accent_prob"].tolist()
        accent_acc = outputs["_accent_acc"].tolist()
        speaker_prob = outputs["speaker_prob"].tolist()
        speaker_acc = outputs["_speaker_acc"].tolist()
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
        parameters = ModelPruning.PARAMETER_NAMES
        current_modules = [(n, m) for n, m in pl_module.model.named_modules()
                           if not isinstance(m, _MODULE_CONTAINERS)]
        parameters_to_prune = [
            (n, m, p) for p in parameters for n, m in current_modules
            if getattr(m, p, None) is not None
        ]

        total_regex = re.compile(
            r"Applied `(?P<prune_fn>\w+)`."
            r" Pruned: (?P<prev_pruned>\d+)/(?P<prev_total>\d+)"
            r" \((?P<prev_prune_rate>\d+\.\d+)\%\)"
            r" -> (?P<curr_pruned>\d+)/(?P<curr_total>\d+)"
            r" \((?P<curr_prune_rate>\d+\.\d+)\%\)"
        )
        module_regex = re.compile(
            r"Applied `(?P<prune_fn>\w+)`"
            r" to `(?P<module>\w+\([()\-\w\s=,.]+\)).(?P<param>\w+)`"
            r" with amount=(?P<amount>\d.\d+)."
            r" Pruned: (?P<prev_pruned>\d+)"
            r" \((?P<prev_prune_rate>\d+\.\d+)\%\)"
            r" -> (?P<curr_pruned>\d+)"
            r" \((?P<curr_prune_rate>\d+\.\d+)\%\)"
        )

        prune_data = []
        epoch = -1
        counter = 0
        pruning_log_file = os.path.join(self.log_dir, "pruning.log")
        with open(pruning_log_file, 'r') as f:
            for line in f:
                result = module_regex.search(line)
                if result is not None:
                    prune_dict = result.groupdict()
                    param = parameters_to_prune[counter-1]
                    for k in prune_dict:
                        if k in {"amount"}:
                            prune_dict[k] = float(prune_dict[k])
                        elif k in {"prev_pruned", "curr_pruned"}:
                            prune_dict[k] = int(prune_dict[k])
                        elif k in {"prev_prune_rate", "curr_prune_rate"}:
                            prune_dict[k] = float(prune_dict[k]) / 100
                        elif k == "module":
                            assert prune_dict[k] == str(param[1])
                            prune_dict[k] = param[0]
                    if hasattr(param[1], param[2]):
                        prev_total = curr_total = getattr(param[1],
                                                          param[2]).numel()
                    prune_data.append({
                        "epoch": epoch,
                        "prev_total": prev_total,
                        "curr_total": curr_total,
                        **prune_dict,
                    })
                else:
                    epoch += 1
                    counter = 0
                    result = total_regex.search(line)
                    prune_dict = result.groupdict()
                    for k in prune_dict:
                        if k in {"amount"}:
                            prune_dict[k] = float(prune_dict[k])
                        if k in {"prev_pruned", "prev_total", "curr_pruned", "curr_total"}:
                            prune_dict[k] = int(prune_dict[k])
                        if k in {"prev_prune_rate", "curr_prune_rate"}:
                            prune_dict[k] = float(prune_dict[k]) / 100
                    prune_data.append({
                        "epoch": epoch,
                        "module": "total",
                        **prune_dict,
                    })
                counter += 1
        df = pd.DataFrame(prune_data)
        print(df)
        print(df.columns)
        for ep in range(epoch+1):
            csv_file_path = os.path.join(self.log_dir, "prune_log",
                                        f"epoch={ep}.csv")
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            df[df["epoch"].values == ep].to_csv(csv_file_path, index=False)
        # print(df[["epoch", "module", "param", "prev_pruned", "prev_total",
        #           "curr_pruned", "curr_total"]].to_string(header=True,
        #                                                   index=True))
        return



def recon_samples(targets, predictions, vocoder, preprocess_config, figure_dir, audio_dir):
    """Reconstruct all samples of the batch."""
    plot_target_specs(targets, predictions, preprocess_config, figure_dir)
    recon_wavs(targets, predictions, vocoder, preprocess_config, audio_dir)

def plot_target_specs(targets, predictions, preprocess_config, figure_dir):
    output = predictions[0]
    src_lens = predictions[8]
    mel_lens = predictions[9]

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

def recon_wavs(targets, predictions, vocoder, preprocess_config, audio_dir,
               postfix=None, save=True):
    # mel_lens = predictions[9]
    # predictions[9] should be the same as targets["mel_lens"]
    mel_lens = targets["mel_lens"]
    postfix = postfix if postfix is not None else ".recon"

    mel_targets = targets["mels"].transpose(1, 2)
    lengths = mel_lens * preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_targets = vocoder.infer(mel_targets, max_wav_value, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    if save:
        for wav, basename in zip(wav_targets, targets["ids"]):
            wav_file_path = os.path.join(audio_dir, f"{basename}/{basename}{postfix}.wav")
            os.makedirs(os.path.dirname(wav_file_path), exist_ok=True)
            wavfile.write(wav_file_path, sampling_rate, wav)
    return wav_targets, targets["ids"], sampling_rate

def synth_samples(targets, predictions, vocoder, preprocess_config, figure_dir, audio_dir, name):
    """Synthesize the first sample of the batch."""
    plot_predict_specs(targets, predictions, preprocess_config, figure_dir, name)
    synth_wavs(targets, predictions, vocoder, preprocess_config, audio_dir, name)

def plot_predict_specs(targets, predictions, preprocess_config, figure_dir, name):
    output = predictions[0]
    postnet_output = predictions[1]
    p_predictions = predictions[2]
    e_predictions = predictions[3]
    d_rounded = predictions[5]
    src_lens = predictions[8]
    mel_lens = predictions[9]

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

def synth_wavs(targets, predictions, vocoder, preprocess_config, audio_dir,
               name=None, postfix=None, save=True, check_recon=True):
    postnet_output = predictions[1]
    mel_lens = predictions[9]
    postfix = postfix if postfix is not None else f".{name}.synth"

    mel_predictions = postnet_output.transpose(1, 2)
    lengths = mel_lens * preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_predictions = vocoder.infer(mel_predictions, max_wav_value, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    if save:
        if check_recon:
            recon = False
            for wav, basename in zip(wav_predictions, targets["ids"]):
                wav_file_path = os.path.join(
                    audio_dir, f"{basename}/{basename}.recon.wav")
                if not os.path.exists(wav_file_path):
                    recon = True
                    break
            if recon:
                recon_wavs(targets, predictions, vocoder, preprocess_config, audio_dir)
        for wav, basename in zip(wav_predictions, targets["ids"]):
            wav_file_path = os.path.join(audio_dir, f"{basename}/{basename}{postfix}.wav")
            os.makedirs(os.path.dirname(wav_file_path), exist_ok=True)
            wavfile.write(wav_file_path, sampling_rate, wav)
    return wav_predictions, targets["ids"], sampling_rate


