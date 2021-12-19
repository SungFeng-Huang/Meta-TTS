import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


SPEAKERS = {
    "french": "css10-fr",
    "german": "css10-de",
    "spanish": "css10-es",
    "russian": "css10-ru",
}


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    # cleaners = config["preprocessing"]["text"]["text_cleaners"]
    path = f'{in_dir}/transcript.txt'
    speaker = SPEAKERS[in_dir.split('/')[-1]]
    tasks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                continue
            wav_name, _, text, _ = line.strip().split('|')
            wav_path = f"{in_dir}/{wav_name}"
            if os.path.isfile(wav_path):
                tasks.append((wav_path, text))
            else:
                print("transcript.txt should not contain non-exist wav files, data might be corrupted.")
                print(f"Can not find {wav_path}.")
    
    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
    for (wav_path, text) in tqdm(tasks):
        base_name = f"{speaker}-{os.path.basename(wav_path).replace('_', '-')[:-4]}"
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
