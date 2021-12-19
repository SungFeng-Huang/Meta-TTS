import os
import glob

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

    tasks = []
    for spk in os.listdir(in_dir):
        if not os.path.isdir(f"{in_dir}/{spk}"):
            continue
        raw_texts = {}
        with open(f"{in_dir}/{spk}/nonpara30/transcripts_utf8.txt", 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                name, raw_text = line.strip().split(":")
                raw_texts[name] = raw_text
        with open(f"{in_dir}/{spk}/parallel100/transcripts_utf8.txt", 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                name, raw_text = line.strip().split(":")
                raw_texts[name] = raw_text
        
        for wav_path in glob.glob(f"{in_dir}/{spk}/nonpara30/wav24kHz16bit/*.wav") + glob.glob(f"{in_dir}/{spk}/parallel100/wav24kHz16bit/*.wav"):
            base_name = os.path.basename(wav_path)[:-4]
            tasks.append((wav_path, raw_texts[base_name], spk))
        os.makedirs(os.path.join(out_dir, spk), exist_ok=True)
    
    for (wav_path, text, spk) in tqdm(tasks):
        base_name = os.path.basename(wav_path)[:-4]
        wav, _ = librosa.load(wav_path, sampling_rate)
        wav = wav / max(abs(wav)) * max_wav_value
        wavfile.write(
            os.path.join(out_dir, spk, "{}.wav".format(base_name)),
            sampling_rate,
            wav.astype(np.int16),
        )
        with open(
            os.path.join(out_dir, spk, "{}.lab".format(base_name)),
            "w",
        ) as f1:
            f1.write(text)
