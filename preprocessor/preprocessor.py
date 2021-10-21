import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
import torch
import s3prl.hub as hub
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.ssl_extractor = getattr(hub, 'hubert_large_ll60k')().cuda()
        self.dvector_extractor = VoiceEncoder()
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        if "subsets" in config:
            self.train_set = config["subsets"].get("train", None)
            self.val_set = config["subsets"].get("val", None)
            self.test_set = config["subsets"].get("test", None)

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "dvector")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "representation")),
                    exist_ok=True)

        print("Processing Data ...")
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers, dvectors = {}, {}
        i = 0   # index of total speakers (train + val + test)
        outs = {}
        for dset in [self.test_set]:
            # for dset in [self.train_set, self.val_set, self.test_set]:
            if dset is None:
                continue
            dset_dir = os.path.join(self.in_dir, dset)
            out = list()
            for speaker in tqdm(os.listdir(dset_dir), desc=dset):
                speakers[speaker] = i
                if speaker not in dvectors:
                    dvectors[speaker] = []
                for wav_name in os.listdir(os.path.join(dset_dir, speaker)):
                    if ".wav" not in wav_name:
                        continue

                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(
                            basename)
                    )
                    if os.path.isfile(tg_path):
                        ret = self.process_utterance(
                            dset_dir, speaker, basename)
                        if ret is None:
                            continue
                        else:
                            info, pitch, energy, dvector, n = ret
                        out.append(info)

                        # dvector shape is (256,)
                        dvectors[speaker].append(dvector)
                        if len(pitch) > 0:
                            pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                        if len(energy) > 0:
                            energy_scaler.partial_fit(energy.reshape((-1, 1)))

                        n_frames += n
                i += 1
            outs[dset] = out

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            # For additional testing corpus/set
            if self.train_set is None and os.path.exists(os.path.join(self.out_dir, "stats.json")):
                stats = json.load(
                    open(os.path.join(self.out_dir, "stats.json"), 'r'))
                pitch_mean = stats['pitch'][2]
                pitch_std = stats['pitch'][3]
            else:
                pitch_mean = pitch_scaler.mean_[0]
                pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            if self.train_set is None and os.path.exists(os.path.join(self.out_dir, "stats.json")):
                stats = json.load(
                    open(os.path.join(self.out_dir, "stats.json"), 'r'))
                energy_mean = stats['energy'][2]
                energy_std = stats['energy'][3]
            else:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f, indent=4)

        for spk, all_dvectors in dvectors.items():
            dvector = np.mean(np.stack(all_dvectors, axis=0), axis=0)
            print(spk)
            print(dvector.shape)
            dvector_filename = "{}-dvector.npy".format(spk)
            np.save(os.path.join(self.out_dir,
                    "dvector", dvector_filename), dvector)

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        # Write metadata
        for dset in outs:
            out = outs[dset]
            with open(os.path.join(self.out_dir, f"{dset}.txt"), "w", encoding="utf-8") as f:
                for m in out:
                    f.write(m + "\n")

        return outs

    def process_utterance(self, in_dir, speaker, basename):
        wav_path = os.path.join(in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, ssl_duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start): int(self.sampling_rate * end)
        ].astype(np.float32)

        # SSL representation
        wav1, _ = librosa.load(wav_path, sr=16000)
        wav1 = wav1[
            int(16000 * start): int(16000 * end)
        ].astype(np.float32)
        wav1 = torch.from_numpy(wav1).float()
        with torch.no_grad():
            representation = self.ssl_extractor(
                [wav1.cuda()])["last_hidden_state"][0]
        representation = representation.detach().cpu().numpy()

        # Speaker dvector extraction
        wav2 = preprocess_wav(Path(wav_path))
        dvector = self.dvector_extractor.embed_utterance(wav2)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64),
                             pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # SSL representation Phoneme-level average
        pos = 0
        for i, d in enumerate(ssl_duration):
            if d > 0:
                representation[i] = np.mean(
                    representation[pos: pos + d], axis=0)
            else:
                representation[i] = np.zeros(1024)
            pos += d
        representation = representation[: len(ssl_duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        representation_filename = "{}-representation-{}.npy".format(
            speaker, basename)
        np.save(os.path.join(self.out_dir, "representation",
                representation_filename), representation)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            dvector,
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations, ssl_durations = [], []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )
            ssl_durations.append(
                int(
                    np.round(e * 50)  # Hubert use 20ms window
                    - np.round(s * 50)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        ssl_durations = ssl_durations[:end_idx]

        return phones, durations, ssl_durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
