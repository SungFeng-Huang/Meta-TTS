import os
from pathlib import Path
from tqdm import tqdm

from dlhlp_lib.text.utils import remove_punctuation

from .KoG2P.g2p import g2p_ko
from src.data.Parsers import get_raw_parser, get_preprocessor


if __name__ == "__main__":
    # ----- kss -----
    raw_dir = Path("/mnt/d/Data/kss")
    preprocessed_dir = Path("preprocessed_data/kss")
    mfa_data_dir = Path("preprocessed_data/kss/mfa_data")
    raw_parser = get_raw_parser("KSS")(raw_dir, preprocessed_dir)
    prepocessor = get_preprocessor("KSS")(preprocessed_dir)
    raw_parser.parse(n_workers=8)
    prepocessor.prepare_mfa(mfa_data_dir)

    # generate dictionary
    dictionary_path = "lexicon/kss-lexicon.txt"
    os.makedirs(os.path.dirname(dictionary_path), exist_ok=True)
    lexicons = {}
    with open(raw_dir / "transcript.v.1.4.txt", 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            if line == "\n":
                continue
            wav, _, text, _, _, _ = line.strip().split("|")
            text = remove_punctuation(text)
            for t in text.split(" "):
                if t not in lexicons:
                    lexicons[t] = g2p_ko(t)

    with open(dictionary_path, 'w', encoding="utf-8") as f:
        for k, v in lexicons.items():
            f.write(f"{k}\t{v}\n")
    print(f"Write {len(lexicons)} words.")
    
    # generate mfa align model
    output_paths = "MFA/Korean/kss_acoustic_model.zip"
    cmd = f"mfa train {str(mfa_data_dir)} {dictionary_path} {output_paths} -j 8 -v --clean"
    os.system(cmd)
