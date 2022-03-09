import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import resemblyzer
from resemblyzer import preprocess_wav, wav_to_mel_spectrogram

import audio as Audio


class Preprocessor:
    def __init__(self, config):
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
        os.makedirs((os.path.join(self.out_dir, "spk_ref_mel_slices")), exist_ok=True)

        print("Processing Data ...")
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        i = 0   # index of total speakers (train + val + test)
        outs = {}
        dsets = []
        for dset in [self.train_set, self.val_set, self.test_set]:
            if dset is None:
                continue
            elif isinstance(dset, list):
                dsets += dset
            elif isinstance(dset, str):
                dsets.append(dset)

        for dset in dsets:
            dset_dir = os.path.join(self.in_dir, dset)
            out = list()
            for speaker in tqdm(os.listdir(dset_dir), desc=dset):
                speakers[speaker] = i
                for wav_name in tqdm(os.listdir(os.path.join(dset_dir, speaker)), leave=False):
                    if ".wav" not in wav_name:
                        continue

                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                    )
                    if os.path.exists(tg_path):
                        ret = self.process_utterance(dset_dir, speaker, basename)
                        if ret is None:
                            continue
                        else:
                            info, pitch, energy, n = ret
                        out.append(info)

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
            # For additional corpus/set
            # if self.train_set is None and os.path.exists(os.path.join(self.out_dir, "stats.json")):
            if os.path.exists(os.path.join(self.out_dir, "stats.json")):
                stats = json.load(open(os.path.join(self.out_dir, "stats.json"), 'r'))
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
            # if self.train_set is None and os.path.exists(os.path.join(self.out_dir, "stats.json")):
            if os.path.exists(os.path.join(self.out_dir, "stats.json")):
                stats = json.load(open(os.path.join(self.out_dir, "stats.json"), 'r'))
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
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    # float(min(pitch_min, -2)),
                    # float(max(pitch_max, 13)),
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    # float(min(energy_min, -1.3)),
                    # float(max(energy_max, 11)),
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
        # phone, duration, start, end = self.get_alignment(
            # textgrid.get_tier_by_name("phones")
        # )
        phone, duration, start, end, word, word_boundary = self.get_word_phone_alignment(
            textgrid.get_tier_by_name("phones"), textgrid.get_tier_by_name("words")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        try:
            pitch, word_pitch = self.extract_pitch(wav, duration, word_boundary)
        except:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy, word_energy = self.extract_melspec_and_energy(
            wav, duration, word_boundary
        )

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)
        if word_pitch is not None:
            pitch_filename = "{}-word-pitch-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "pitch", pitch_filename), word_pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)
        if word_energy is not None:
            energy_filename = "{}-word-energy-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "energy", energy_filename), word_energy)


        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        # speaker d-vector reference
        # spk_ref_mel_slices = self.get_mel_slices_for_d_vec(wav_path)
        # spk_ref_mel_slices_filename = mel_filename
        # np.save(
            # os.path.join(self.out_dir, "spk_ref_mel_slices", spk_ref_mel_slices_filename),
            # spk_ref_mel_slices,
        # )

        if "<unk>" in [t.text for t in textgrid.get_tier_by_name("words")]:
            # At least still save files, just not include all.txt
            return None

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
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

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

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

    def get_word_phone_alignment(self, p_tier, w_tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0

        words = []
        word_se = []
        word_durations = []
        word_it = iter(w_tier)
        w = next(word_it)
        ws, we, wt = w.start_time, w.end_time, w.text

        for t in p_tier._objects:
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

            # Map to corresponding word
            # print(p, s, e, wt, ws, we)
            while we < e:
                try:
                    w = next(word_it)
                    ws, we, wt = w.start_time, w.end_time, w.text
                except:
                    ws, we, wt = s, e, "[SEP]"

            if ws <= s and we >= e:
                words.append(wt)
                word_se.append([
                    int(
                        np.round(ws * self.sampling_rate / self.hop_length)
                        - np.round(start_time * self.sampling_rate / self.hop_length)
                    ),
                    int(
                        np.round(we * self.sampling_rate / self.hop_length)
                        - np.round(start_time * self.sampling_rate / self.hop_length)
                    ),
                ])
                # word_durations.append(
                    # int(
                        # np.round(we * self.sampling_rate / self.hop_length)
                        # - np.round(ws * self.sampling_rate / self.hop_length)
                    # )
                # )
            elif s < ws:
                # For silent phones
                assert p in sil_phones
                words.append("[SEP]")
                word_se.append([
                    int(
                        np.round(s * self.sampling_rate / self.hop_length)
                        - np.round(start_time * self.sampling_rate / self.hop_length)
                    ),
                    int(
                        np.round(e * self.sampling_rate / self.hop_length)
                        - np.round(start_time * self.sampling_rate / self.hop_length)
                    ),
                ])
                # word_durations.append(durations[-1])

            try:
                assert word_se[-1][0] <= word_se[-1][1]
            except Exception as e:
                print(t, w)
                raise e

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        word_se = word_se[:end_idx]
        for i in range(end_idx):
            try:
                assert word_se[i][0] <= min(word_se[i][1], sum(durations)-1)
                word_se[i][1] = min(word_se[i][1], sum(durations)-1)
            except Exception as e:
                print(phones[i], durations[i], word_se[i], sum(durations))
                raise e
        # end_duration = int(np.round(end_time * self.sampling_rate / self.hop_length)
                           # - np.round(start_time * self.sampling_rate / self.hop_length))
        # for s, e in word_se[:end_idx]:
            # if e > end_frame:
                # word_durations.append(end_frame - s)
            # else:
                # word_durations.append(e - s)

        return phones, durations, start_time, end_time, words, word_se

    def get_mel_slices_for_d_vec(self, wav_path):
        # Settings are slightly different from above, so should start again
        wav = preprocess_wav(wav_path)

        # Compute where to split the utterance into partials and pad the waveform
        # with zeros if the partial utterances cover a larger range.
        wav_slices, mel_slices = resemblyzer.VoiceEncoder.compute_partial_slices(
            len(wav), rate=1.3, min_coverage=0.75
        )
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        # Split the utterance into partials and forward them through the model
        spk_ref_mel = wav_to_mel_spectrogram(wav)
        spk_ref_mel_slices = [spk_ref_mel[s] for s in mel_slices]
        return spk_ref_mel_slices

    def extract_pitch(self, wav, duration, word_boundary=None):
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        word_pitch = None
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

            # Word-level average
            if word_boundary is not None:
                word_pitch = np.empty_like(pitch[: len(duration)])
                for i, (s, e) in enumerate(word_boundary):
                    if s < e:
                        word_pitch[i] = np.mean(pitch[s:e])
                    else:
                        word_pitch[i] = 0

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        return pitch, word_pitch

    def extract_melspec_and_energy(self, wav, duration, word_boundary=None):
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        word_energy = None
        if self.energy_phoneme_averaging:
            # Word-level average
            if word_boundary is not None:
                word_energy = np.empty_like(energy[: len(duration)])
                for i, (s, e) in enumerate(word_boundary):
                    if s < e:
                        word_energy[i] = np.mean(energy[s:e])
                    else:
                        word_energy[i] = 0

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]
            
        return mel_spectrogram, energy, word_energy
