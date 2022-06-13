import json
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, prosody_averaging, merge_stats, merge_speaker_map
from .tts import TTSDataset


class MonolingualTTSDataset(TTSDataset):
    def __init__(
        self, dset, preprocess_config, train_config,
        spk_refer_wav=False
    ):
        super().__init__(dset, preprocess_config, train_config, spk_refer_wav)
        self.lang_id = preprocess_config["lang_id"]

        if "df_path" in preprocess_config["path"]:
            df_path = preprocess_config["path"]["df_path"]
        else:
            df_path = "preprocessed_data/VCTK-speaker-info.csv"
        self.df = pd.read_csv(df_path).fillna("Unknown")
        self.speaker_accent_map = {
            data["pID"]: data["ACCENTS"] for _, data in self.df.iterrows()
        }
        self.speaker_region_map = {
            data["pID"]: (data["ACCENTS"], data["REGION"])
            for _, data in self.df.iterrows()
        }
        self.accent_map = {
            k: i for i, k in enumerate(self.df.groupby(["ACCENTS"]).groups)
        }
        self.region_map = {
            k: i for i, k in
            enumerate(self.df.groupby(["ACCENTS", "REGION"]).groups)
        }
        self.accent = [self.speaker_accent_map.get(spk, None)
                       for spk in self.speaker]
        self.region = [self.speaker_region_map.get(spk, None)
                       for spk in self.speaker]

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

        accent = self.accent[idx]
        region = self.region[idx]

        if accent is not None and region is not None:
            accent_id = self.accent_map[accent]
            region_id = self.region_map[region]
            sample.update({
                "accent": accent_id,
                "region": region_id,
            })

        return sample
