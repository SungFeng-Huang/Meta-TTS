import os
import glob
import numpy as np

from text import text_to_sequence
from lightning.datamodules.define import LANG_ID2SYMBOLS


preprocessed_path = "./preprocessed_data/LibriTTS"
filename = "./preprocessed_data/LibriTTS/train-clean-100.txt"
lang_id = 0
output_path = "./preprocessed_data/LibriTTS/train-clean-100_phoneme-features.npy"

with open(filename, "r", encoding="utf-8") as f:
    names = []
    speakers = []
    texts = []
    raw_texts = []
    for line in f.readlines():
        n, s, t, r = line.strip("\n").split("|")
        names.append(n)
        speakers.append(s)
        texts.append(t)
        raw_texts.append(r)

d_representation = 1024
n_symbols = len(LANG_ID2SYMBOLS[lang_id])
for idx in range(len(names)):
    phone = np.array(text_to_sequence(texts[idx], ["english_cleaners"]))
    representation_path = os.path.join(
        preprocessed_path,
        "representation",
        "{}-representation-{}.npy".format(speakers[idx], names[idx]),
    )
    representation = np.load(representation_path)

    table = {i: np.zeros(d_representation) for i in range(n_symbols)}
    cnt = {i: 0 for i in range(n_symbols)}
    print(len(phone), len(representation))
    print(phone)
    for ph, r in zip(phone, representation):
        array_has_nan = np.isnan(np.sum(r))
        if array_has_nan:
            print(names[idx])
            r[np.isnan(r)] = 0
        table[int(ph)] += r
        cnt[int(ph)] += 1

phoneme_features = np.zeros((n_symbols, d_representation), dtype=np.float32)
rr = []
for i in range(n_symbols):
    if cnt[i] == 0:
        phoneme_features[i] = np.zeros(d_representation)
    else:
        rr.append(i)
        phoneme_features[i] = table[int(i)] / cnt[i]

print(phoneme_features.shape)
print(rr)
with open(output_path, 'wb') as f:
    np.save(f, phoneme_features)
