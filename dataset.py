import json
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, prosody_averaging, merge_stats, merge_speaker_map


class XvecDataset(Dataset):
    df = pd.read_csv("preprocessed_data/VCTK-speaker-info.csv").fillna("Unknown")
    speaker_accent_map = {
        data["pID"]: data["ACCENTS"] for _, data in df.iterrows()
    }
    speaker_region_map = {
        data["pID"]: (data["ACCENTS"], data["REGION"])
        for _, data in df.iterrows()
    }
    accent_map = {k: i for i, k in enumerate(df.groupby(["ACCENTS"]).groups)}
    region_map = {
        k: i for i, k in enumerate(df.groupby(["ACCENTS", "REGION"]).groups)
    }

    def __init__(self, dset, preprocess_config, train_config):
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, _, _ = self.process_meta(
            dset
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.accent = [self.speaker_accent_map[spk] for spk in self.speaker]
        self.region = [self.speaker_region_map[spk] for spk in self.speaker]

        print(f"\nLength of dataset: {len(self.speaker)}")

    def __len__(self):
        return len(self.speaker)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        accent = self.accent[idx]
        region = self.region[idx]

        speaker_id = self.speaker_map[speaker]
        accent_id = self.accent_map[accent]
        region_id = self.region_map[region]

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "accent": accent_id,
            "region": region_id,
            "mel": mel,
        }
        return sample

    def process_meta(self, dset):
        name = []
        speaker = []
        text = []
        raw_text = []
        with open(
            os.path.join(self.preprocessed_path, f"{dset}.txt"), "r", encoding="utf-8"
        ) as f:
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
        return name, speaker, text, raw_text


class TTSDataset(Dataset):
    def __init__(
        self, dset, preprocess_config, train_config,
        spk_refer_wav=False
    ):
        super().__init__()
        self.dset = dset
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.spk_refer_wav = spk_refer_wav
        # if spk_refer_wav:
        # self.raw_path = preprocess_config["path"]["raw_path"]

        self.pitch_phoneme_averaging = (
            preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        self.pitch_log = preprocess_config["preprocessing"]["pitch"]["log"]
        self.energy_log = preprocess_config["preprocessing"]["energy"]["log"]
        self.pitch_normalization = (
            preprocess_config["preprocessing"]["pitch"]["normalization"]
        )
        self.energy_normalization = (
            preprocess_config["preprocessing"]["energy"]["normalization"]
        )

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            dset
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.set_speaker_map(json.load(f))
        if self.pitch_normalization or self.energy_normalization:
            with open(os.path.join(self.preprocessed_path, "stats.json")) as f:
                self.set_stats(json.load(f))


        print(f"\nLength of dataset: {len(self.text)}")

    def set_stats(self, stats):
        self.stats = stats

    def merge_stats(self, other_stats):
        if getattr(self, "stats", None) is None:
            self.stats = other_stats
            return
        self.stats = merge_stats(self.stats, other_stats)

    def set_speaker_map(self, speaker_map):
        self.speaker_map = speaker_map

    def merge_speaker_map(self, speaker_map):
        if getattr(self, "speaker_map", None) is None:
            self.speaker_map = speaker_map
            return
        self.speaker_map = merge_speaker_map(self.speaker_map, speaker_map)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)

        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        if self.pitch_phoneme_averaging:
            pitch = prosody_averaging(pitch, duration, interp=True)
        if self.pitch_log:
            pitch = np.log(pitch + 1e-12)
        elif self.pitch_normalization:
            pitch = (pitch - self.stats["pitch"]["mean"]) / self.stats["pitch"]["std"]

        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        if self.energy_phoneme_averaging:
            energy = prosody_averaging(energy, duration)
        if self.energy_log:
            energy = np.log(energy + 1e-12)
        elif self.energy_normalization:
            energy = (energy - self.stats["energy"]["mean"]) / self.stats["energy"]["std"]

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        if self.spk_refer_wav:
            spk_ref_mel_slices_path = os.path.join(
                self.preprocessed_path,
                "spk_ref_mel_slices",
                "{}-mel-{}.npy".format(speaker, basename),
            )
            spk_ref_mel_slices = np.load(spk_ref_mel_slices_path)

            sample.update({"spk_ref_mel_slices": spk_ref_mel_slices})

        return sample

    def process_meta(self, dset):
        name = []
        speaker = []
        text = []
        raw_text = []
        with open(
            os.path.join(self.preprocessed_path, f"{dset}.txt"), "r", encoding="utf-8"
        ) as f:
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
        return name, speaker, text, raw_text


class MonolingualTTSDataset(TTSDataset):
    def __init__(
        self, dset, preprocess_config, train_config,
        spk_refer_wav=False
    ):
        super().__init__(dset, preprocess_config, train_config, spk_refer_wav)
        self.lang_id = preprocess_config["lang_id"]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        basename = self.basename[idx]
        speaker = self.speaker[idx]

        representation_path = os.path.join(
            self.preprocessed_path,
            "representation",
            "{}-representation-{}.npy".format(speaker, basename),
        )
        if not os.path.isfile(representation_path):
            representation = np.zeros((1, 1024))
        else:
            representation = np.load(representation_path)

        sample.update({
            "language": self.lang_id,
            "representation": representation,
        })

        return sample


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


