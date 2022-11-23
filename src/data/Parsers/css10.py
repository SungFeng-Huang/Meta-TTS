import os
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool

from dlhlp_lib.tts_preprocess.utils import ImapWrapper
from dlhlp_lib.tts_preprocess.basic import *
from dlhlp_lib.text.utils import lowercase

import Define
from Parsers.interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from . import template


SPEAKERS = {
    "french": "css10-fr",
    "german": "css10-de",
    "spanish": "css10-es",
    "dutch": "css10-nl",
    "russian": "css10-ru",
}


class CSS10RawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.lang = str(self.root).split('/')[-1]
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_initial_features(self, query, data):
        template.prepare_initial_features(self.data_parser, query, data)

    def parse(self, n_workers=4):
        res = {"data": [], "data_info": [], "all_speakers": []}
        path = self.root / "transcript.txt"
        speaker = SPEAKERS[self.lang]
        res["all_speakers"].append(speaker)
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line == '\n':
                    continue
                wav_name, _, text, _ = line.strip().split('|')
                wav_path = self.root / wav_name
                basename = wav_name.split('/')[-1][:-4]
                if os.path.isfile(wav_path):
                    data = {
                        "wav_path": wav_path,
                        "text": text,
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": f"{speaker}-{basename}",
                    }
                    res["data"].append(data)
                    res["data_info"].append(data_info)
                else:
                    print("transcript.txt should not contain non-exist wav files, data might be corrupted.")
                    print(f"Can not find {wav_path}.")

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


class CSS10Preprocessor(BasePreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root)
        self.lang = str(self.root).split('/')[-1]
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_mfa(self, mfa_data_dir: Path):
        queries = self.data_parser.get_all_queries()
        self.data_parser.text.read_all()

        # 1. create a similar structure in "mfa_data_dir" as in "wav_dir"
        for spk in self.data_parser.get_all_speakers():
            target_dir = mfa_data_dir / spk
            target_dir.mkdir(parents=True, exist_ok=True)

        # 2. create hard link for wav file
        for query in tqdm(queries):
            target_dir = mfa_data_dir / query['spk']
            link_file = target_dir / f"{query['basename']}.wav"
            txt_link_file = target_dir / f"{query['basename']}.txt"
            wav_file = self.data_parser.wav_16000.read_filename(query, raw=True)
            txt = self.data_parser.text.read_from_query(query)
            
            if link_file.exists():
                os.unlink(str(link_file))
            os.link(wav_file, str(link_file))
            with open(txt_link_file, 'w', encoding="utf-8") as f:
                f.write(lowercase(txt))

    def mfa(self, mfa_data_dir: Path):
        corpus_directory = str(mfa_data_dir)
        dictionary_path = f"lexicon/{SPEAKERS[self.lang]}-lexicon.txt"
        acoustic_model_path = f"MFA/{self.lang.capitalize()}/{SPEAKERS[self.lang]}_acoustic_model.zip"
        output_directory = str(self.root / "TextGrid")
        cmd = f"mfa align {corpus_directory} {dictionary_path} {acoustic_model_path} {output_directory} -j 8 -v --clean"
        os.system(cmd)
    
    def denoise(self):
        pass

    def preprocess(self):
        queries = self.data_parser.get_all_queries()
        if Define.DEBUG:
            queries = queries[:128]
        template.preprocess(self.data_parser, queries)

    def split_dataset(self, cleaned_data_info_path: str):
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        template.split_monospeaker_dataset(self.data_parser, queries, output_dir, val_size=1000)
