import sys
import os
import pytest

from torch.testing import assert_allclose, assert_close
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer


import torch
import librosa
from librosa.core.spectrum import _spectrogram
import src.audio as Audio
import yaml
import numpy as np
import torchaudio.transforms as T

config = yaml.load(open("config/preprocess/VCTK.yaml", 'r'),
                   Loader=yaml.FullLoader)
wav1, _ = librosa.load('raw_data/VCTK/Indian/Unknown/p251/p251_001.wav')
wav2, _ = librosa.load('raw_data/VCTK/Indian/Unknown/p251/p251_002.wav')
wav3, _ = librosa.load('raw_data/VCTK/Indian/Unknown/p251/p251_003.wav')

class SrcAudioSTFT:
    def __init__(self, config) -> None:
        self.config = config
        self.stft = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def magnitude(self, wav):
        mag, phase = self.stft.stft_fn.transform(torch.FloatTensor(wav).unsqueeze(0))
        mag = torch.squeeze(mag, 0)#numpy().astype(np.float32)
        return mag
    
    def energy(self, wav):
        logmelspec, energy = Audio.tools.get_mel_from_wav(wav, self.stft)
        return energy

    def logmelspec(self, wav):
        logmelspec, energy = Audio.tools.get_mel_from_wav(wav, self.stft)
        return logmelspec

    def pitch(self, wav):
        config = self.config
        import pyworld as pw
        sample_rate=config["preprocessing"]["audio"]["sampling_rate"]
        hop_length=config["preprocessing"]["stft"]["hop_length"]
        pitch, t = pw.dio(
            wav.astype(np.float64),
            sample_rate,
            frame_period=hop_length/sample_rate*1000,
        )
        print(pitch, pitch.shape)
        pitch=pw.stonemask(
            wav.astype(np.float64),
            pitch,
            t,
            sample_rate,
        )
        print(pitch, pitch.shape)


class LibrosaSTFT:
    def __init__(self, config) -> None:
        self.config = config

    def magnitude(self, wav):
        config = self.config
        S = np.abs(
            librosa.stft(
                wav,
                n_fft=config["preprocessing"]["stft"]["filter_length"],
                hop_length=config["preprocessing"]["stft"]["hop_length"],
                win_length=config["preprocessing"]["stft"]["win_length"],
                center=True,
                window='hann',
                pad_mode='reflect',
            )
        ) ** 1 #power
        return S

    def energy(self, wav):
        S = self.magnitude(wav)
        return torch.norm(torch.tensor(S), dim=-2)
    
    def logmelspec(self, wav):
        config = self.config
        S = self.magnitude(wav)
        # return librosa.power_to_db(S, ref=1.0)
        libmelspec = librosa.feature.melspectrogram(
            wav,
            sr=config["preprocessing"]["audio"]["sampling_rate"],
            S=S,# if pre-computed
            n_fft=config["preprocessing"]["stft"]["filter_length"],
            hop_length=config["preprocessing"]["stft"]["hop_length"],
            win_length=config["preprocessing"]["stft"]["win_length"],
            power=1,
            pad_mode="reflect",
            n_mels=config["preprocessing"]["mel"]["n_mel_channels"],
            fmin=config["preprocessing"]["mel"]["mel_fmin"],
            fmax=config["preprocessing"]["mel"]["mel_fmax"],
        )
        liblogmelspec = Audio.audio_processing.dynamic_range_compression(torch.FloatTensor(libmelspec))
        return liblogmelspec


class TorchAudioSTFT:
    def __init__(self, config) -> None:
        self.config = config
        self.t = T.MelSpectrogram(
            sample_rate=config["preprocessing"]["audio"]["sampling_rate"],
            n_fft=config["preprocessing"]["stft"]["filter_length"],
            hop_length=config["preprocessing"]["stft"]["hop_length"],
            win_length=config["preprocessing"]["stft"]["win_length"],
            center=True,
            pad_mode="reflect",
            power=1,
            n_mels=config["preprocessing"]["mel"]["n_mel_channels"],
            norm="slaney",
            mel_scale="slaney",
        )

    def magnitude(self, wav):
        tmag = self.t.spectrogram(torch.FloatTensor(wav))
        return tmag

    def energy(self, wav):
        tmag = self.magnitude(wav)
        return torch.norm(tmag, dim=-2)
    
    def logmelspec(self, wav):
        tmelspec = self.t(torch.FloatTensor(wav))
        tlogmelspec = Audio.audio_processing.dynamic_range_compression(tmelspec)
        return tlogmelspec

    def pitch(self, wav):
        # would not be the same as pyworld output
        config = self.config
        sample_rate=config["preprocessing"]["audio"]["sampling_rate"]
        hop_length=config["preprocessing"]["stft"]["hop_length"]

        from torchaudio import functional as F
        pitch = F.detect_pitch_frequency(torch.FloatTensor(wav), sample_rate)
        print(pitch, pitch.shape)
        pitch = F.compute_kaldi_pitch(torch.FloatTensor(wav), sample_rate)
        pitch, nfcc = pitch[..., 0], pitch[..., 1]
        print(pitch, pitch.shape)


src_stft = SrcAudioSTFT(config)
librosa_stft = LibrosaSTFT(config)
torchaudio_stft = TorchAudioSTFT(config)


@pytest.mark.parametrize("wav", [wav1, wav2, wav3])
def test_magnitud(wav):
    # Magnitude

    # src.audio version
    mag = src_stft.magnitude(wav)

    # librosa version
    S = librosa_stft.magnitude(wav)
    assert_allclose(S, mag)

    # torchaudio version
    tmag = torchaudio_stft.magnitude(wav)
    assert_allclose(tmag, mag)


@pytest.mark.parametrize("wav", [wav1, wav2, wav3])
def test_energy(wav):
    # Energy

    # src.audio version
    energy = src_stft.energy(wav)

    # librosa version
    # libenergy = torch.norm(torch.tensor(S).unsqueeze(0), dim=-2)
    libenergy = librosa_stft.energy(wav)
    assert_allclose(libenergy, energy)

    # torchaudio version
    tenergy = torchaudio_stft.energy(wav)
    assert_allclose(tenergy, energy)


@pytest.mark.parametrize("wav", [wav1, wav2, wav3])
def test_logmelspec(wav):
    # LogMelSpec

    # src.audio version
    logmelspec = src_stft.logmelspec(wav)

    # librosa version
    liblogmelspec = librosa_stft.logmelspec(wav)
    assert_allclose(liblogmelspec, logmelspec)

    # torchaudio version
    tlogmelspec = torchaudio_stft.logmelspec(wav)
    assert_allclose(tlogmelspec, logmelspec)


@pytest.mark.parametrize("wavs", [[wav1, wav2, wav3]])
def test_batch_process(wavs: list):
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    win_length = config["preprocessing"]["stft"]["win_length"]

    spec_lens = [len(w) // hop_length + 1 for w in wavs]

    _wavs = torch.nn.utils.rnn.pad_sequence(
        [
            torch.cat([
                torch.FloatTensor(w),
                torch.FloatTensor(w).flip([0])[1:win_length]
            ])
            for w in wavs
        ],
        batch_first=True
    )

    tmags = torchaudio_stft.magnitude(_wavs)
    tenergies = torchaudio_stft.energy(_wavs)
    tlogmelspecs = torchaudio_stft.logmelspec(_wavs)

    for i, wav in enumerate(wavs):
        mag = src_stft.magnitude(wav)
        energy = src_stft.energy(wav)
        logmelspec = src_stft.logmelspec(wav)

        assert mag.shape[1] == spec_lens[i]

        assert_allclose(tmags[i, :, :spec_lens[i]], mag)
        assert_allclose(tenergies[i, :spec_lens[i]], energy)
        assert_allclose(tlogmelspecs[i, :, :spec_lens[i]], logmelspec)



