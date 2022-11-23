import os
from pathlib import Path

from src.data.Parsers import get_raw_parser, get_preprocessor


def german_scripts():
    raw_dir = Path("/mnt/d/Data/CSS10/german")
    preprocessed_dir = Path("preprocessed_data/CSS10/german")
    mfa_data_dir = Path("preprocessed_data/CSS10/german/mfa_data")
    raw_parser = get_raw_parser("CSS10")(raw_dir, preprocessed_dir)
    prepocessor = get_preprocessor("CSS10")(preprocessed_dir)
    raw_parser.parse(n_workers=8)
    prepocessor.prepare_mfa(mfa_data_dir)

    # generate dictionary
    dictionary_path = "lexicon/css10-de-lexicon.txt"
    os.system("mfa models download g2p german_mfa")
    os.system(f"mfa g2p german_mfa {str(mfa_data_dir)} {dictionary_path} --clean")
    
    # generate mfa align model
    os.system("mfa models download acoustic german_mfa")
    output_paths = "MFA/German/css10-de_acoustic_model.zip"
    cmd = f"mfa adapt {str(mfa_data_dir)} {dictionary_path} german_mfa {output_paths} -j 8 -v --clean"
    os.system(cmd)


def french_scripts():
    raw_dir = Path("/mnt/d/Data/CSS10/french")
    preprocessed_dir = Path("preprocessed_data/CSS10/french")
    mfa_data_dir = Path("preprocessed_data/CSS10/french/mfa_data")
    raw_parser = get_raw_parser("CSS10")(raw_dir, preprocessed_dir)
    prepocessor = get_preprocessor("CSS10")(preprocessed_dir)
    raw_parser.parse(n_workers=8)
    prepocessor.prepare_mfa(mfa_data_dir)

    # generate dictionary
    dictionary_path = "lexicon/css10-fr-lexicon.txt"
    os.system("mfa models download g2p french_mfa")
    os.system(f"mfa g2p french_mfa {str(mfa_data_dir)} {dictionary_path} --clean")
    
    # generate mfa align model
    os.system("mfa models download acoustic french_mfa")
    output_paths = "MFA/French/css10-fr_acoustic_model.zip"
    cmd = f"mfa adapt {str(mfa_data_dir)} {dictionary_path} french_mfa {output_paths} -j 8 -v --clean"
    os.system(cmd)


def spanish_scripts():
    raw_dir = Path("/mnt/d/Data/CSS10/spanish")
    preprocessed_dir = Path("preprocessed_data/CSS10/spanish")
    mfa_data_dir = Path("preprocessed_data/CSS10/spanish/mfa_data")
    raw_parser = get_raw_parser("CSS10")(raw_dir, preprocessed_dir)
    prepocessor = get_preprocessor("CSS10")(preprocessed_dir)
    raw_parser.parse(n_workers=8)
    prepocessor.prepare_mfa(mfa_data_dir)

    # generate dictionary
    dictionary_path = "lexicon/css10-es-lexicon.txt"
    os.system("mfa models download g2p spanish_spain_mfa")
    os.system(f"mfa g2p spanish_spain_mfa {str(mfa_data_dir)} {dictionary_path} --clean")
    
    # generate mfa align model
    os.system("mfa models download acoustic spanish_mfa")
    output_paths = "MFA/Spanish/css10-es_acoustic_model.zip"
    cmd = f"mfa adapt {str(mfa_data_dir)} {dictionary_path} spanish_mfa {output_paths} -j 8 -v --clean"
    os.system(cmd)


def dutch_scripts():
    # TODO
    raw_dir = Path("/mnt/d/Data/CSS10/dutch")
    preprocessed_dir = Path("preprocessed_data/CSS10/dutch")
    mfa_data_dir = Path("preprocessed_data/CSS10/dutch/mfa_data")
    raw_parser = get_raw_parser("CSS10")(raw_dir, preprocessed_dir)
    prepocessor = get_preprocessor("CSS10")(preprocessed_dir)
    raw_parser.parse(n_workers=8)
    prepocessor.prepare_mfa(mfa_data_dir)

    # download dictionary
    os.system("mfa models download dictionary dutch_cv")
    
#     # generate mfa align model
#     output_paths = "MFA/Dutch/css10-nl_acoustic_model.zip"
#     cmd = f"mfa train {str(mfa_data_dir)} {dictionary_path} {output_paths} -j 8 -v --clean"
#     os.system(cmd)


if __name__ == "__main__":
    french_scripts()
    german_scripts()
    spanish_scripts()
    # dutch_scripts()
