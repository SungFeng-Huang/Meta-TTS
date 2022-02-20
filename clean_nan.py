import os
import numpy as np
import json
import random
import shutil
from tqdm import tqdm


MAX_FRAME = 10000
TASKS = {
    # "preprocessed_data/GlobalPhone/es": {
    #     "train": "preprocessed_data/GlobalPhone/es/train.txt",
    #     "val": "preprocessed_data/GlobalPhone/es/val.txt",
    #     "test": "preprocessed_data/GlobalPhone/es/val.txt",
    # },
    # "preprocessed_data/JVS": {
    #     "train": "preprocessed_data/JVS/train.txt",
    #     "val": "preprocessed_data/JVS/val.txt",
    #     "test": "preprocessed_data/JVS/val.txt",
    # },
    # "preprocessed_data/LibriTTS": {
    #     "train": "preprocessed_data/LibriTTS/train-clean-100.txt",
    #     "val": "preprocessed_data/LibriTTS/dev-clean.txt",
    #     "test": "preprocessed_data/LibriTTS/test-clean.txt",
    # },
    # "preprocessed_data/AISHELL-3": {
    #     "train": "preprocessed_data/AISHELL-3/train.txt",
    #     "val": "preprocessed_data/AISHELL-3/val.txt",
    #     "test": "preprocessed_data/AISHELL-3/val.txt",
    # },
    # "preprocessed_data/GlobalPhone/fr": {
    #     "train": "preprocessed_data/GlobalPhone/fr/train.txt",
    #     "val": "preprocessed_data/GlobalPhone/fr/val.txt",
    #     "test": "preprocessed_data/GlobalPhone/fr/val.txt",
    # },
    # "preprocessed_data/GlobalPhone/de": {
    #     "train": "preprocessed_data/GlobalPhone/de/train.txt",
    #     "val": "preprocessed_data/GlobalPhone/de/val.txt",
    #     "test": "preprocessed_data/GlobalPhone/de/val.txt",
    # },
    # "preprocessed_data/GlobalPhone/cz": {
    #     "train": "preprocessed_data/GlobalPhone/cz/train.txt",
    #     "val": "preprocessed_data/GlobalPhone/cz/val.txt",
    #     "test": "preprocessed_data/GlobalPhone/cz/val.txt",
    # },
    "preprocessed_data/CSS10/german": {
        "train": "preprocessed_data/CSS10/german/train.txt",
        "val": "preprocessed_data/CSS10/german/val.txt",
        "test": "preprocessed_data/CSS10/german/val.txt",
    },
}


for (corpus_path, sets) in TASKS.items():
    time_info = {}
    for set_name, info_path in sets.items():
        lines = []
        time_info[set_name] = 0
        with open(info_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                wav_name, spk, phns, raw_text = line.strip().split("|")
                lines.append((spk, line))
        lines = [l[1] for l in lines]
        checked = []
        
        for line in tqdm(lines):
            wav_name, spk, phns, raw_text = line.strip().split("|")
            to_check = [
                f"representation/{spk}-representation-{wav_name}.npy",
                f"pitch/{spk}-pitch-{wav_name}.npy",
                f"duration/{spk}-duration-{wav_name}.npy",
                f"energy/{spk}-energy-{wav_name}.npy",
                f"mel/{spk}-mel-{wav_name}.npy",
                f"spk_ref_mel_slices/{spk}-mel-{wav_name}.npy",
            ]
            ok = True
            for idx, filename in enumerate(to_check):
                arr = np.load(f"{corpus_path}/{filename}")
                if np.any(np.isnan(arr)):
                    print(f"NaN detected on {corpus_path}/{filename}.")
                    ok = False
                    break
                if idx == 4 and arr.shape[0] > MAX_FRAME:
                    ok = False
                    break
            if ok:
                checked.append(line)
                mel = np.load(f"{corpus_path}/{to_check[4]}")
                assert mel.shape[1] == 80
                time_info[set_name] += mel.shape[0] * 256 / 22050 / 60  # minutes
        
        with open(f"{info_path[:-4]}-clean.txt", 'w', encoding='utf-8') as f:
            for line in checked:
                f.write(f"{line}")

    # Write time information
    with open(f"{corpus_path}/info.json", 'w', encoding='utf-8') as f:
        json.dump(time_info, f, indent=4)

    print(f"Remove NaN from {corpus_path}, done.")
    print("")
