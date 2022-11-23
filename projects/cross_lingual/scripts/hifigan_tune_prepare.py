import os
from tqdm import tqdm
import random

from src.data.Parsers.parser import DataParser


MAX_SAMPLE_PER_DATASET = 10000


def main(roots, output_dir):
    # Create hard links from prepocessed data
    all_data = []
    for root in roots:
        print(f"Linking mels and wavs from {root}...")
        data_parser = DataParser(root)
        data_parser.text.read_all()
        queries = data_parser.get_all_queries()
        if len(queries) > MAX_SAMPLE_PER_DATASET:
            queries = random.sample(queries, MAX_SAMPLE_PER_DATASET)
        for query in tqdm(queries):
            mel_src = data_parser.mel.read_filename(query, raw=True)
            wav_src = data_parser.wav_trim_22050.read_filename(query, raw=True)
            try:
                assert os.path.exists(mel_src) and os.path.exists(wav_src)
            except:
                continue
            mel_dst = f"{output_dir}/data/{query['basename']}.npy"
            wav_dst = f"{output_dir}/data/{query['basename']}.wav"
            if os.path.exists(mel_dst):
                os.unlink(mel_dst)
            if os.path.exists(wav_dst):
                os.unlink(wav_dst)
            os.link(mel_src, mel_dst)
            os.link(wav_src, wav_dst)

            text = data_parser.text.read_from_query(query)
            all_data.append((query["basename"], text))

    # Create train/val split for hifigan tuning
    random.shuffle(all_data)
    with open(f"{output_dir}/train.txt", 'w', encoding="utf-8") as f:
        for (n, t) in all_data[:-150]:
            f.write(f"{n}|{t}\n")
    with open(f"{output_dir}/val.txt", 'w', encoding="utf-8") as f:
        for (n, t) in all_data[-150:]:
            f.write(f"{n}|{t}\n")


if __name__ == "__main__":
    output_dir = "preprocessed_data/hifigan"
    roots = [
        "preprocessed_data/LJSpeech",
        "preprocessed_data/LibriTTS",
        "preprocessed_data/AISHELL-3",
        "preprocessed_data/JSUT",
        "preprocessed_data/kss",
        "preprocessed_data/CSS10/german",
        "preprocessed_data/CSS10/spanish",
        "preprocessed_data/CSS10/french"
    ]

    random.seed(666)
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    main(roots, output_dir)
