import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    # cleaners = config["preprocessing"]["text"]["text_cleaners"]
    path = f'{in_dir}/train/label_train-set.txt'
    tasks = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5 or line == '\n':
                continue
            wav_name, text, _ = line.strip().split('|')
            wav_path = f"{in_dir}/train/wav/{wav_name[:-4]}/{wav_name}.wav"
            speaker = wav_name[:-4]
            if os.path.isfile(wav_path):
                tasks.append((wav_path, text, speaker))
            else:
                print("label_train-set.txt should not contain non-exist wav files, data might be corrupted.")
                print(f"Can not find {wav_path}.")
    
    for (wav_path, text, speaker) in tqdm(tasks):
        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        base_name = os.path.basename(wav_path)[:-4]
        wav, _ = librosa.load(wav_path, sampling_rate)
        wav = wav / max(abs(wav)) * max_wav_value
        wavfile.write(
            os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
            sampling_rate,
            wav.astype(np.int16),
        )
        with open(
            os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
            "w",
        ) as f1:
            f1.write(text)
