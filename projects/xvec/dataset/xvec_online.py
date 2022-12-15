import json
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .xvec import XvecDataset
from src.data.Parser.parser import DataParser


class XvecOnlineDataset(XvecDataset):

    def __init__(self, dset, data_parser: DataParser, preprocess_config: dict):
        super().__init__(dset, preprocess_config)

        if data_parser is None:
            data_parser = DataParser(self.preprocessed_path)
        self.data_parser = data_parser

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        accent = self.accent[idx]
        region = self.region[idx]

        speaker_id = self.speaker_map[speaker]
        accent_id = self.accent_map[accent]
        region_id = self.region_map[region]

        query = {
            "spk": speaker,
            "basename": basename,
        }
        wav = self.data_parser.wav.read_from_query(query)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "accent": accent_id,
            "region": region_id,
            "wav": wav,
        }
        return sample

