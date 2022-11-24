import os
import glob
import json
from typing import Dict, List

from dlhlp_lib.parsers.Interfaces import BaseDataParser, BaseQueryParser
from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import SFQueryParser, NestSFQueryParser
from dlhlp_lib.parsers.IOObjects import NumpyIO, PickleIO, WavIO, TextGridIO, TextIO, JSONIO


class SBQueryParser(BaseQueryParser):
    """
    [root]/[spk]-[basename][extension]
    """
    def __init__(self, root):
        super().__init__(root)
    
    def get(self, query) -> List[str]:
        return [f"{query['spk']}-{query['basename']}"]

    def get_all(self, extension) -> List[str]:
        return [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(f"{self.root}/*{extension}")]

    def get_cache(self) -> str:
        return f"{self.root}/cache.pickle"


class DataParser(BaseDataParser):
    def __init__(self, root):
        super().__init__(root)

        self.wav = Feature(
            SBQueryParser(f"{self.root}/wav"), WavIO(sr=22050))
        self.mel = Feature(
            SBQueryParser(f"{self.root}/mel"), NumpyIO())
        self.pitch = Feature(
            SBQueryParser(f"{self.root}/pitch"), NumpyIO(), enable_cache=True)
        self.energy = Feature(
            SBQueryParser(f"{self.root}/energy"), NumpyIO(), enable_cache=True)
        self.duration = Feature(
            SBQueryParser(f"{self.root}/duration"), NumpyIO(), enable_cache=True)
        
        self.textgrid = Feature(
            NestSFQueryParser(f"{self.root}/TextGrid"), TextGridIO())
        self.phoneme = Feature(
            SBQueryParser(f"{self.root}/phoneme"), TextIO(), enable_cache=True)
        self.text = Feature(
            SBQueryParser(f"{self.root}/text"), TextIO(), enable_cache=True)
        self.spk_ref_mel_slices = Feature(
            SBQueryParser(f"{self.root}/spk_ref_mel_slices"), NumpyIO())
        
        self.stats_path = f"{self.root}/stats.json"
        self.speakers_path = f"{self.root}/speakers.json"
        self.metadata_path = f"{self.root}/data_info.json"

    def _init_structure(self):
        os.makedirs(f"{self.root}/wav", exist_ok=True)
        os.makedirs(f"{self.root}/text", exist_ok=True)
    
    def get_all_queries(self):
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            data_infos = json.load(f)
        return data_infos
    
    def get_feature(self, query: str) -> Feature:
        return getattr(self, query)
    
    def get_all_speakers(self) -> List[str]:
        with open(self.speakers_path, 'r', encoding='utf-8') as f:
            speakers = json.load(f)
        return speakers
