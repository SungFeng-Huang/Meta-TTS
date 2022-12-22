import os

from .xvec import XvecDataset
from src.data.Parsers.parser import DataParser


class XvecOnlineDataset(XvecDataset):
    """
    Changes:
        - speaker map: change from dict to list
    """

    def __init__(self, dset, preprocess_config: dict,
                 data_parser: DataParser = None):
        super().__init__(dset, preprocess_config)

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
        filename = os.path.join(self.preprocessed_path, f"{dset}.txt")
        with open(filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                n, s, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
        return name, speaker
