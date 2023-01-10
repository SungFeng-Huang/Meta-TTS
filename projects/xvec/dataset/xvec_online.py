import os
import pandas as pd

from .xvec import XvecDataset
from src.data.Parsers.parser import DataParser


class XvecOnlineDataset(XvecDataset):
    """
    Changes:
        - speaker map: change from dict to list
    """

    def __init__(self, corpus_id, dset, path_config: dict,
                 data_parser: DataParser = None):
        """
        spk_id depends on "speakers.json"
        accent_id, region_id depends on "./preprocessed_data/VCTK-speaker-info.csv"

        Args:
            dset: (list of) subset name | subset filename
                load from "path_config["preprocessed_path"]/{dset}.txt"
            path_config:
                dictionary consist at least 2 keys:
                - df_path: "./preprocessed_data/VCTK-speaker-info.csv"
                - preprocessed_path
        """
        super(XvecDataset, self).__init__()

        self.corpus_id = corpus_id
        self.preprocessed_path = path_config["preprocessed_path"]

        # load VCTK speaker-accent info
        self.df = pd.read_csv(path_config["df_path"]).fillna("Unknown")

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

        if data_parser is None:
            data_parser = DataParser(self.preprocessed_path)
        self.data_parser = data_parser

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        accent = self.accent[idx]
        region = self.region[idx]

        speaker_id = self.speaker_map.index(speaker)
        accent_id = self.accent_map[accent]
        region_id = self.region_map[region]

        query = {
            "spk": speaker,
            "basename": basename,
        }
        wav = self.data_parser.wav.read_from_query(query)

        sample = {
            "corpus": self.corpus_id,
            "id": basename,
            "speaker": speaker_id,
            "accent": accent_id,
            "region": region_id,
            "wav": wav,
        }
        return sample

    def process_meta(self, dset):
        name = []
        speaker = []

        if isinstance(dset, list):
            for ds in dset:
                n, s = self.process_meta(ds)
                name += n
                speaker += s
            return name, speaker

        assert isinstance(dset, str)
        if os.path.exists(dset):
            filename = dset
        else:
            filename = os.path.join(self.preprocessed_path, f"{dset}.txt")
        with open(filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                n, s, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
        return name, speaker
