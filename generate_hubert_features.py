import os
import glob
import numpy as np
from tqdm import tqdm

from text import text_to_sequence
from text.define import LANG_ID2SYMBOLS


def generate_hubert_features(filename, lang_id):
    preprocessed_path = os.path.dirname(filename)
    output_path = f"{os.path.dirname(filename)}/{os.path.basename(filename)[:-4]}_phoneme-features.npy"

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
    table = {i: np.zeros(d_representation) for i in range(n_symbols)}
    cnt = {i: 0 for i in range(n_symbols)}
    for idx in tqdm(range(len(names))):
        phone = np.array(text_to_sequence(texts[idx], ["basic_cleaners"], lang_id))
        representation_path = os.path.join(
            preprocessed_path,
            "representation",
            "{}-representation-{}.npy".format(speakers[idx], names[idx]),
        )
        representation = np.load(representation_path)

        for ph, r in zip(phone, representation):
            array_has_nan = np.isnan(np.sum(r))
            if array_has_nan:
                print(names[idx], ph)
                continue
            table[int(ph)] += r
            cnt[int(ph)] += 1

    phoneme_features = np.zeros((n_symbols, d_representation), dtype=np.float32)
    rr = []
    for i in range(n_symbols):
        if cnt[i] == 0:
            phoneme_features[i] = np.zeros(d_representation)
        else:
            rr.append(i)
            phoneme_features[i] = table[i] / cnt[i]

    with open(output_path, 'wb') as f:
        np.save(f, phoneme_features)
    print(phoneme_features.shape)
    print([LANG_ID2SYMBOLS[lang_id][r] for r in rr])


if __name__ == "__main__":
    generate_hubert_features("./preprocessed_data/LibriTTS/train-clean-100.txt", lang_id=0)
    generate_hubert_features("./preprocessed_data/AISHELL-3/train.txt", lang_id=1)
    # generate_hubert_features("./preprocessed_data/GlobalPhone/cz/train.txt", lang_id=7)
    # generate_hubert_features("./preprocessed_data/GlobalPhone/es/train.txt", lang_id=5)
    # generate_hubert_features("./preprocessed_data/GlobalPhone/fr/train.txt", lang_id=2)
    # generate_hubert_features("./preprocessed_data/GlobalPhone/de/train.txt", lang_id=3)
