import os
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool

from dlhlp_lib.tts_preprocess.utils import ImapWrapper
from dlhlp_lib.tts_preprocess.basic import *

import Define
from .interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from .utils import write_queries_to_txt
from . import template


class LibriTTSRawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.data_parser = DataParser(str(preprocessed_root))
        self.dsets = [
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ]

    def prepare_initial_features(self, query, data):
        template.prepare_initial_features(self.data_parser, query, data)

    def parse(self, n_workers=4):
        res = {"data": [], "data_info": [], "all_speakers": []}
        for dset in self.dsets:
            if not os.path.isdir(f"{self.root}/{dset}"):
                continue
            for speaker in tqdm(os.listdir(f"{self.root}/{dset}"), desc=dset):
                res["all_speakers"].append(speaker)
                for chapter in os.listdir(f"{self.root}/{dset}/{speaker}"):
                    for filename in os.listdir(f"{self.root}/{dset}/{speaker}/{chapter}"):
                        if filename[-4:] != ".wav":
                            continue
                        basename = filename[:-4]
                        wav_path = f"{self.root}/{dset}/{speaker}/{chapter}/{basename}.wav"
                        text_path = f"{self.root}/{dset}/{speaker}/{chapter}/{basename}.normalized.txt"
                        with open(text_path, "r", encoding="utf-8") as f:
                            text = f.readline().strip("\n")
                        data = {
                            "wav_path": wav_path,
                            "text": text,
                        }
                        data_info = {
                            "spk": speaker,
                            "basename": basename,
                            "dset": dset,
                            "chapter": chapter,
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


class LibriTTSPreprocessor(BasePreprocessor):
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
        train_set, dev_set, test_set = [], [], []
        for q in queries:
            if "train" in q["dset"]:
                train_set.append(q)
            elif "dev" in q["dset"]:
                dev_set.append(q)
            elif "test" in q["dset"]:
                test_set.append(q)
        
        write_queries_to_txt(self.data_parser, train_set, f"{output_dir}/train.txt")
        write_queries_to_txt(self.data_parser, dev_set, f"{output_dir}/val.txt")
        write_queries_to_txt(self.data_parser, test_set, f"{output_dir}/test.txt")
