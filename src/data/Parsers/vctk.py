import os
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool
import csv
import pandas as pd

from dlhlp_lib.tts_preprocess.utils import ImapWrapper
from dlhlp_lib.tts_preprocess.basic import *

import Define
from .interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from .utils import write_queries_to_txt
from . import template


class VCTKRawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.preprocessed_root = preprocessed_root
        self.data_parser = DataParser(str(preprocessed_root))

        self.data_parser.speaker_info_path = f"{preprocessed_root}/speaker-info.csv"
        if not os.path.exists(self.data_parser.speaker_info_path):
            self.parse_speaker_info()
        self.speaker_info = pd.read_csv(self.data_parser.speaker_info_path).fillna("Unknown")
        self.speaker_accent_map = {}    # spk -> accent
        self.speaker_region_map = {}    # spk -> region
        for _, data in self.speaker_info.iterrows():
            spk, accent = data["ID"], data["ACCENTS"]
            region = (data["ACCENTS"], data["REGION"])
            self.speaker_accent_map[spk] = accent
            self.speaker_region_map[spk] = region

    def parse_speaker_info(self):
        with open(os.path.join(self.root, "speaker-info.txt"), 'r') as f:
            lines = [l.strip().split(maxsplit=4) for l in f.readlines()]
            lines[0][4] = lines[0][4].split()[0]
            for l in lines:
                if l[0] in ["p280", "p315"]:
                    # mic2 unavailable
                    l[4] = l[4][:l[4].index('(')].strip()   # remove "(...unavailable)"
                    continue
                elif l[0] == "s5":
                    l[4] = ""
                elif l[3] == "Australian":
                    # ACCENT        REGION          =>  ACCENT              REGION
                    # Australian    English Sydney  =>  AustralianEnglish   Sydney
                    # Australian    English         =>  AustralianEnglish
                    assert len(l) == 5
                    assert l[4].split()[0] == "English"
                    l[3] = "AustralianEnglish"
                    if len(l[4].split()) > 1:
                        l[4] = l[4].split(maxsplit=1)[1]
                    else:
                        l[4] = ""
                elif len(l) == 4:
                    l.append("")

        with open(self.data_parser.speaker_info_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(lines)

    def prepare_initial_features(self, query, data):
        template.prepare_initial_features(self.data_parser, query, data)

    def parse(self, n_workers=4):
        res = {"data": [], "data_info": [], "all_speakers": []}
        for speaker in tqdm(os.listdir(f"{self.root}/wav48_silence_trimmed"),
                            desc="speakers"):
            if os.path.isfile(f"{self.root}/wav48_silence_trimmed/{speaker}"):
                # it's "log.txt", not a speaker
                continue
            res["all_speakers"].append(speaker)
            for filename in os.listdir(f"{self.root}/wav48_silence_trimmed/{speaker}"):
                if filename[-10:] != "_mic2.flac":
                    continue
                basename = filename[:-10]
                wav_path = f"{self.root}/wav48_silence_trimmed/{speaker}/{basename}_mic2.flac"
                text_path = f"{self.root}/txt/{speaker}/{basename}.txt"
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.readline().strip("\n")

                data = {
                    "wav_path": wav_path,
                    "text": text,
                }
                data_info = {
                    "spk": speaker,
                    "basename": basename,
                    "accent": self.speaker_accent_map[speaker],
                    "region": self.speaker_region_map[speaker],
                }
                res["data"].append(data)
                res["data_info"].append(data_info)

        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(res["data_info"], f, indent=4)
        with open(self.data_parser.speakers_path, "w", encoding="utf-8") as f:
            json.dump(res["all_speakers"], f, indent=4)

        n = len(res["data_info"])
        tasks = list(zip(res["data_info"], res["data"], [False] * n))
        
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(ImapWrapper(self.prepare_initial_features), tasks, chunksize=64), total=n):
                pass
        self.data_parser.text.read_all(refresh=True)


class VCTKPreprocessor(BasePreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root)
        self.data_parser = DataParser(str(preprocessed_root))

    # Use prepared textgrids from ming024's repo
    def prepare_mfa(self, mfa_data_dir: Path):
        pass
    
    # Use prepared textgrids from ming024's repo
    def mfa(self, mfa_data_dir: Path):
        pass
    
    def denoise(self):
        pass

    def preprocess(self):
        # queries = self.data_parser.get_all_queries()
        # if Define.DEBUG:
        #     queries = queries[:128]
        # template.preprocess(self.data_parser, queries)
        pass

    def split_dataset(self, cleaned_data_info_path: str):
        output_dir = os.path.dirname(cleaned_data_info_path)
        # with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
        #     queries = json.load(f)
        # template.split_multispeaker_dataset(self.data_parser, queries, output_dir, val_spk_size=40)

        queries = self.data_parser.get_all_queries()
        accents = {}
        # train_set, dev_set, test_set = [], [], []
        for q in queries:
            if q["accent"] not in accents:
                accents[q["accent"]] = []
            accents[q["accent"]].append(q)

        for accent in accents:
            write_queries_to_txt(self.data_parser, accents[accent],
                                 f"{output_dir}/{accent}.txt")
        write_queries_to_txt(self.data_parser, sum(accents.values(), []),
                             f"{output_dir}/total.txt")

