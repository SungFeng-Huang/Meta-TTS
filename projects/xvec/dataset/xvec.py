import json
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class XvecDataset(Dataset):

    def __init__(self, dset, preprocess_config):
        """
        spk_id depends on "speakers.json"
        accent_id, region_id depends on "./preprocessed_data/VCTK-speaker-info.csv"

        Args:
            dset: load from "preprocess_config["path"]["preprocessed_path"]/{dset}.txt"
            preprocess_config:
                dictionary consist at least a key "path":
                - path:
                    dictionary consist at least 2 keys:
                    - df_path: "./preprocessed_data/VCTK-speaker-info.csv"
                    - preprocessed_path
        """
        super().__init__()

        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]

        # load VCTK speaker-accent info
        self.df = pd.read_csv(preprocess_config["path"]["df_path"]).fillna("Unknown")

        # build maps
        self.build_speaker2id()
        self.build_accent2id()
        self.build_region2id()
        self.build_speaker2accent()
        self.build_speaker2region()

        # create list of basename and speaker
        self.basename, self.speaker = self.process_meta(dset)

        # create list of accent and region
        self.accent, self.region = [], []
        for spk in self.speaker:
            self.accent.append(self.speaker_accent_map.get(spk, None))
            self.region.append(self.speaker_region_map.get(spk, None))

        # add None to accent and region map
        if None in self.accent:
            self.accent_map[None] = -100
        if None in self.region:
            self.region_map[None] = -100

        print(f"\nLength of dataset: {len(self.speaker)}")

    def build_speaker2id(self):
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

    def build_accent2id(self):
        self.accent_map = {}
        for i, k in enumerate(self.df.groupby(["ACCENTS"]).groups):
            self.accent_map[k] = i

    def build_region2id(self):
        self.region_map = {}
        for i, k in enumerate(self.df.groupby(["ACCENTS", "REGION"]).groups):
            self.region_map[k] = i

    def build_speaker2accent(self):
        self.speaker_accent_map = {}    # spk -> accent
        for _, data in self.df.iterrows():
            spk, accent = data["ID"], data["ACCENTS"]
            self.speaker_accent_map[spk] = accent

    def build_speaker2region(self):
        self.speaker_region_map = {}    # spk -> region
        for _, data in self.df.iterrows():
            spk = data["ID"]
            region = (data["ACCENTS"], data["REGION"])
            self.speaker_region_map[spk] = region

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
        filename = os.path.join(self.preprocessed_path, f"{dset}.txt")
        with open(filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
        return name, speaker
