import json
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class XvecDataset(Dataset):

    def __init__(self, dset, preprocess_config):
        """
        Args:
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

        # load speaker-id map
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

        # build accent-id and region-id map
        self.accent_map = {}
        for i, k in enumerate(self.df.groupby(["ACCENTS"]).groups):
            self.accent_map[k] = i
        self.region_map = {}
        for i, k in enumerate(self.df.groupby(["ACCENTS", "REGION"]).groups):
            self.region_map[k] = i

        # build spk->accent and spk->region maps
        self.speaker_accent_map = {}    # spk -> accent
        self.speaker_region_map = {}    # spk -> region
        for _, data in self.df.iterrows():
            spk, accent = data["pID"], data["ACCENTS"]
            region = (data["ACCENTS"], data["region"])
            self.speaker_accent_map[spk] = accent
            self.speaker_region_map[spk] = region

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
        with open(
            os.path.join(self.preprocessed_path, f"{dset}.txt"), "r", encoding="utf-8"
        ) as f:
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
        return name, speaker
