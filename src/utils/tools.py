import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml(yaml_in: str) -> Dict[str, Any]:
    return yaml.load(open(yaml_in, 'r'), Loader=yaml.FullLoader)


def to_device(data, device):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[11][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
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

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        # wav_reconstruction = wav_prediction = None
        return fig, mel_target, mel_prediction, basename

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path, suffix=None, figname='Synthesized Spectrogram'):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            [figname],
        )
        if suffix is not None:
            basename = f"{basename}.{suffix}"
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    if vocoder is not None:
        from .model import vocoder_infer

        mel_predictions = predictions[1].transpose(1, 2)
        lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
        wav_predictions = vocoder_infer(
            mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
        )

        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        for wav, basename in zip(wav_predictions, basenames):
            wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min = stats["pitch"]["min"]
    pitch_max = stats["pitch"]["max"]
    energy_min = stats["energy"]["min"]
    energy_max = stats["energy"]["max"]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    fig.subplots_adjust(hspace=0.3)
    for i in range(len(data)):
        mel, pitch, energy = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0, mode="constant"):
    def pad_data(x, length, PAD):
        kwargs = {}
        if mode == "constant":
            kwargs = {"constant_values": PAD}
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode=mode, **kwargs
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def prosody_averaging(prosody, duration, interp=False, word_boundary=None):
    # change zeros into interpolate values
    if interp:
        # perform linear interpolation
        nonzero_ids = np.where(prosody != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            prosody[nonzero_ids],
            fill_value=(prosody[nonzero_ids[0]], prosody[nonzero_ids[-1]]),
            bounds_error=False,
        )
        prosody = interp_fn(np.arange(0, len(prosody)))

    # Phoneme-level average
    pos = 0
    phoneme_prosody = np.empty_like(prosody[: len(duration)])
    for i, d in enumerate(duration):
        if d > 0:
            phoneme_prosody[i] = np.mean(prosody[pos : pos + d])
        else:
            phoneme_prosody[i] = 0
        pos += d

    # Word-level average
    if word_boundary is not None:
        word_prosody = np.empty_like(prosody[: len(duration)])
        for i, (s, e) in enumerate(word_boundary):
            if s < e:
                word_prosody[i] = np.mean(prosody[s:e])
            else:
                word_prosody[i] = 0
        return phoneme_prosody, word_prosody

    return phoneme_prosody


def merge_stats(origin_stats, *other_stats):
    new_stats = origin_stats
    for stats in other_stats:
        old_stats = new_stats
        new_stats = {}
        for k in ["pitch", "energy"]:
            M = old_stats[k]["n_samples"]
            N = stats[k]["n_samples"]
            mu_m = old_stats[k]["mean"]
            mu_n = stats[k]["mean"]
            var_m = old_stats[k]["std"] ** 2
            var_n = stats[k]["std"] ** 2
            new_stats[k] = {
                "n_samples": M + N,
                "min": min(old_stats[k]["min"], stats[k]["min"]),
                "max": max(old_stats[k]["max"], stats[k]["max"]),
                "mean": (M*mu_m + N*mu_n) / (M + N),
                "std": np.sqrt(
                    (M*var_m + N*var_n + M*N/(M+N)*((mu_m-mu_n)**2)) / (M + N)
                ),
            }
    return new_stats


def merge_speaker_map(*speaker_maps):
    new_speaker_map = {}
    i = 0
    for spk_map in speaker_maps:
        for spk in spk_map:
            new_speaker_map[spk] = i
            i += 1
    return new_speaker_map
