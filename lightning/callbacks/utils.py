import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile

from utils.tools import expand, plot_mel


def synth_one_sample_with_target(targets, predictions, vocoder, preprocess_config):
    """Synthesize the first sample of the batch given target pitch/duration/energy."""
    basename = targets[0][0]
    src_len         = predictions[8][0].item()
    mel_len         = predictions[9][0].item()
    mel_target      = targets[6][0, :mel_len].detach().transpose(0, 1)
    duration        = targets[11][0, :src_len].detach().cpu().numpy()
    pitch           = targets[9][0, :src_len].detach().cpu().numpy()
    energy          = targets[10][0, :src_len].detach().cpu().numpy()
    mel_prediction  = predictions[1][0, :mel_len].detach().transpose(0, 1)
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

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

def recon_samples(targets, predictions, vocoder, preprocess_config, figure_dir, audio_dir):
    """Reconstruct all samples of the batch."""
    for i in range(len(predictions[0])):
        basename    = targets[0][i]
        src_len     = predictions[8][i].item()
        mel_len     = predictions[9][i].item()
        mel_target  = targets[6][i, :mel_len].detach().transpose(0, 1)
        duration    = targets[11][i, :src_len].detach().cpu().numpy()
        pitch       = targets[9][i, :src_len].detach().cpu().numpy()
        energy      = targets[10][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        else:
            pitch = targets[9][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        else:
            energy = targets[10][i, :mel_len].detach().cpu().numpy()

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

    mel_targets = targets[6].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_targets = vocoder.infer(mel_targets, max_wav_value, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_targets, targets[0]):
        wavfile.write(os.path.join(audio_dir, f"{basename}.recon.wav"), sampling_rate, wav)

def synth_samples(targets, predictions, vocoder, preprocess_config, figure_dir, audio_dir, name):
    """Synthesize the first sample of the batch."""
    for i in range(len(predictions[0])):
        basename        = targets[0][i]
        src_len         = predictions[8][i].item()
        mel_len         = predictions[9][i].item()
        mel_prediction  = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration        = predictions[5][i, :src_len].detach().cpu().numpy()
        pitch           = predictions[2][i, :src_len].detach().cpu().numpy()
        energy          = predictions[3][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        else:
            pitch = targets[9][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        else:
            energy = targets[10][i, :mel_len].detach().cpu().numpy()

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

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_predictions = vocoder.infer(mel_predictions, max_wav_value, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, targets[0]):
        wavfile.write(os.path.join(audio_dir, f"{basename}.{name}.synth.wav"), sampling_rate, wav)

