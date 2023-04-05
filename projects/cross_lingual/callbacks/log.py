import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.vocoders import BaseVocoder

import Define
from cross_lingual.utils.tool import expand, plot_mel


def synth_one_sample_with_target(targets, predictions, vocoder: BaseVocoder, model_config):
    """Synthesize the first sample of the batch given target pitch/duration/energy."""
    basename = targets[0][0]
    src_len         = predictions[8][0].item()
    mel_len         = predictions[9][0].item()
    mel_target      = targets[6][0, :mel_len].detach().transpose(0, 1)
    duration        = targets[11][0, :src_len].detach().cpu().numpy()
    pitch           = targets[9][0, :src_len].detach().cpu().numpy()
    energy          = targets[10][0, :src_len].detach().cpu().numpy()
    mel_prediction  = predictions[1][0, :mel_len].detach().transpose(0, 1)
    if model_config["pitch"]["feature"] == "phoneme_level":
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if model_config["energy"]["feature"] == "phoneme_level":
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    # Reconstruct raw pitch/energy
    _, _, pitch_mu, pitch_std, _, _, energy_mu, energy_std = Define.ALLSTATS["global"]
    if model_config["pitch"]["normalization"]:
        pitch = pitch * pitch_std + pitch_mu
    if model_config["pitch"]["normalization"]:
        energy = energy * energy_std + energy_mu
    
    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        Define.ALLSTATS["global"],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )
    wav_reconstruction = vocoder.infer(mel_target.unsqueeze(0))[0]
    wav_prediction = vocoder.infer(mel_prediction.unsqueeze(0))[0]

    return fig, wav_reconstruction, wav_prediction, basename


def recon_samples(targets, predictions, vocoder: BaseVocoder, model_config, figure_dir, audio_dir):
    """Reconstruct all samples of the batch."""
    for i in range(len(predictions[0])):
        basename    = targets[0][i]
        src_len     = predictions[8][i].item()
        mel_len     = predictions[9][i].item()
        mel_target  = targets[6][i, :mel_len].detach().transpose(0, 1)
        duration    = targets[11][i, :src_len].detach().cpu().numpy()
        pitch       = targets[9][i, :src_len].detach().cpu().numpy()
        energy      = targets[10][i, :src_len].detach().cpu().numpy()
        if model_config["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        else:
            pitch = targets[9][i, :mel_len].detach().cpu().numpy()
        if model_config["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        else:
            energy = targets[10][i, :mel_len].detach().cpu().numpy()

        # Reconstruct raw pitch/energy
        _, _, pitch_mu, pitch_std, _, _, energy_mu, energy_std = Define.ALLSTATS["global"]
        if model_config["pitch"]["normalization"]:
            pitch = pitch * pitch_std + pitch_mu
        if model_config["pitch"]["normalization"]:
            energy = energy * energy_std + energy_mu

        fig = plot_mel(
            [
                (mel_target.cpu().numpy(), pitch, energy),
            ],
            Define.ALLSTATS["global"],
            ["Ground-Truth Spectrogram"],
        )
        plt.savefig(os.path.join(figure_dir, f"{basename}.target.png"))
        plt.close()

    mel_targets = targets[6].transpose(1, 2)
    lengths = predictions[9] * AUDIO_CONFIG["stft"]["hop_length"]
    wav_targets = vocoder.infer(mel_targets, lengths=lengths)

    sampling_rate = AUDIO_CONFIG["audio"]["sampling_rate"]
    for wav, basename in zip(wav_targets, targets[0]):
        wavfile.write(os.path.join(audio_dir, f"{basename}.recon.wav"), sampling_rate, wav)


def synth_samples(targets, predictions, vocoder: BaseVocoder, model_config, figure_dir, audio_dir, name):
    """Synthesize the first sample of the batch."""
    for i in range(len(predictions[0])):
        basename        = targets[0][i]
        src_len         = predictions[8][i].item()
        mel_len         = predictions[9][i].item()
        mel_prediction  = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration        = predictions[5][i, :src_len].detach().cpu().numpy()
        pitch           = predictions[2][i, :src_len].detach().cpu().numpy()
        energy          = predictions[3][i, :src_len].detach().cpu().numpy()
        if model_config["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        else:
            pitch = targets[9][i, :mel_len].detach().cpu().numpy()
        if model_config["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        else:
            energy = targets[10][i, :mel_len].detach().cpu().numpy()

        # Reconstruct raw pitch/energy
        _, _, pitch_mu, pitch_std, _, _, energy_mu, energy_std = Define.ALLSTATS["global"]
        if model_config["pitch"]["normalization"]:
            pitch = pitch * pitch_std + pitch_mu
        if model_config["pitch"]["normalization"]:
            energy = energy * energy_std + energy_mu

        with open(os.path.join(figure_dir, f"{basename}.{name}.synth.npy"), 'wb') as f:
            np.save(f, mel_prediction.cpu().numpy())
        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            Define.ALLSTATS["global"],
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(figure_dir, f"{basename}.{name}.synth.png"))
        plt.close()

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * AUDIO_CONFIG["stft"]["hop_length"]
    try:
        wav_predictions = vocoder.infer(mel_predictions, lengths=lengths)

        sampling_rate = AUDIO_CONFIG["audio"]["sampling_rate"]
        for wav, basename in zip(wav_predictions, targets[0]):
            wavfile.write(os.path.join(audio_dir, f"{basename}.{name}.synth.wav"), sampling_rate, wav)
    except:
        print("Vocoder fails, if happened in early training stage, might due to very very short mel spectrograms.")
