import json
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class XvecDataset(Dataset):

    def __init__(self, dset, preprocess_config):
        super().__init__()
        self.df = pd.read_csv(preprocess_config["path"]["df_path"]).fillna("Unknown")
        self.speaker_accent_map = {
            data["pID"]: data["ACCENTS"] for _, data in self.df.iterrows()
        }
        self.speaker_region_map = {
            data["pID"]: (data["ACCENTS"], data["REGION"])
            for _, data in self.df.iterrows()
        }

        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]

        self.basename, self.speaker, _, _ = self.process_meta(
            dset
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.accent = [self.speaker_accent_map.get(spk, None)
                       for spk in self.speaker]
        self.region = [self.speaker_region_map.get(spk, None)
                       for spk in self.speaker]
        self.accent_map = {
            None: -100,
            **{k: i for i, k in enumerate(self.df.groupby(["ACCENTS"]).groups)}
        }
        self.region_map = {
            None: -100,
            **{k: i for i, k in
               enumerate(self.df.groupby(["ACCENTS", "REGION"]).groups)}
        }

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


