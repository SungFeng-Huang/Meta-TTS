import os
import json
import random
import shutil
from tqdm import tqdm


SPKPERCORPUS = 5
DATAPERSPK = 20
TASKS = {
    # "miniLibriTTS": {
    #     "train": "preprocessed_data/LibriTTS/train-clean-100.txt",
    #     "val": "preprocessed_data/LibriTTS/dev-clean.txt",
    #     "test": "preprocessed_data/LibriTTS/test-clean.txt",
    # },
    # "miniVCTK": "preprocessed_data/VCTK/all.txt",
    "miniAISHELL-3": {
        "train": "preprocessed_data/AISHELL-3/train.txt",
        "val": "preprocessed_data/AISHELL-3/val.txt",
        "test": "preprocessed_data/AISHELL-3/val.txt",
    },

    # "miniCV-french": "preprocessed_data/CommonVoice/fr/train.txt",
    # "miniCV-german": "preprocessed_data/CommonVoice/de/train.txt",
    # "miniCV-russian": "preprocessed_data/CommonVoice/ru/train.txt",

    # "miniCSS10-french": "preprocessed_data/CSS10/fr/train.txt",
    # "miniCSS10-german": "preprocessed_data/CSS10/de/train.txt",
    # "miniCSS10-spanish": "preprocessed_data/CSS10/es/train.txt",
    # "miniCSS10-russian": "preprocessed_data/CSS10/ru/train.txt",
    
    # "miniJVS": "JVS/train.txt",

    # "miniGlobalPhone-fr": {
    #     "train": "preprocessed_data/GlobalPhone/fr/train.txt",
    #     "val": "preprocessed_data/GlobalPhone/fr/train.txt",
    #     "test": "preprocessed_data/GlobalPhone/fr/train.txt",
    # },
    # "miniGlobalPhone-de": {
    #     "train": "preprocessed_data/GlobalPhone/de/train.txt",
    #     "val": "preprocessed_data/GlobalPhone/de/train.txt",
    #     "test": "preprocessed_data/GlobalPhone/de/train.txt",
    # },
    # "miniGlobalPhone-es": {
    #     "train": "preprocessed_data/GlobalPhone/es/train.txt",
    #     "val": "preprocessed_data/GlobalPhone/es/train.txt",
    #     "test": "preprocessed_data/GlobalPhone/es/train.txt",
    # },
    # "miniGlobalPhone-cz": {
    #     "train": "preprocessed_data/GlobalPhone/cz/train.txt",
    #     "val": "preprocessed_data/GlobalPhone/cz/train.txt",
    #     "test": "preprocessed_data/GlobalPhone/cz/train.txt",
    # },
}

for corpus_name, sets in TASKS.items():
    shutil.rmtree(f"preprocessed_data/{corpus_name}", ignore_errors=True)  # clear mini corpus
    mini_corpus_path = f"preprocessed_data/{corpus_name}"
    os.makedirs(mini_corpus_path, exist_ok=True)
    os.makedirs(f"{mini_corpus_path}/representation", exist_ok=True)
    os.makedirs(f"{mini_corpus_path}/duration", exist_ok=True)
    os.makedirs(f"{mini_corpus_path}/pitch", exist_ok=True)
    os.makedirs(f"{mini_corpus_path}/energy", exist_ok=True)
    os.makedirs(f"{mini_corpus_path}/mel", exist_ok=True)
    os.makedirs(f"{mini_corpus_path}/spk_ref_mel_slices", exist_ok=True)
    os.makedirs(f"{mini_corpus_path}/TextGrid", exist_ok=True)

    for set_name, info_path in sets.items():
        full_corpus_path = os.path.dirname(info_path)

        # choose speakers
        with open(f"{full_corpus_path}/speakers.json", "r", encoding='utf-8') as f:
            speakers = json.load(f)
        candidates = {x: [] for x in speakers}
        with open(info_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                wav_name, spk, phns, raw_text = line.strip().split("|")
                candidates[spk].append(line)
        exist_spks = [spk for spk, lines in candidates.items() if len(lines) > 0]
        if len(exist_spks) < SPKPERCORPUS:
            chosen_speakers = exist_spks
        else:
            chosen_speakers = random.sample(exist_spks, k=SPKPERCORPUS)
        for spk in chosen_speakers:
            os.makedirs(f"{mini_corpus_path}/TextGrid/{spk}", exist_ok=True)

        # choose tasks
        mini_corpus_tasks = []
        for spk, lines in candidates.items():
            if spk in chosen_speakers:
                print(f"Sample {DATAPERSPK} from {len(lines)}...")
                if len(lines) < DATAPERSPK:
                    mini_corpus_tasks.extend(lines)
                else:
                    mini_corpus_tasks.extend(random.sample(lines, k=DATAPERSPK))
        
        # create mini corpus
        print(f"Create preprocessed_data/{corpus_name}/{set_name}.txt...")
        with open(f"preprocessed_data/{corpus_name}/{set_name}.txt", 'w', encoding='utf-8') as f:
            for line in mini_corpus_tasks:
                f.write(line)
        shutil.copyfile(f"{full_corpus_path}/speakers.json", f"{mini_corpus_path}/speakers.json")
        shutil.copyfile(f"{full_corpus_path}/stats.json", f"{mini_corpus_path}/stats.json")
        for line in tqdm(mini_corpus_tasks):
            wav_name, spk, phns, raw_text = line.strip().split("|")
            to_copy = [
                f"TextGrid/{spk}/{wav_name}.TextGrid",
                f"representation/{spk}-representation-{wav_name}.npy",
                f"pitch/{spk}-pitch-{wav_name}.npy",
                f"duration/{spk}-duration-{wav_name}.npy",
                f"energy/{spk}-energy-{wav_name}.npy",
                f"mel/{spk}-mel-{wav_name}.npy",
                f"spk_ref_mel_slices/{spk}-mel-{wav_name}.npy",
            ]
            for filename in to_copy:
                shutil.copyfile(f"{full_corpus_path}/{filename}", f"{mini_corpus_path}/{filename}")
