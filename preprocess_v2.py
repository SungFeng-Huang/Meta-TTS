import argparse
import os
import torchaudio
from pathlib import Path

import Define
from src.data.Parsers import get_raw_parser, get_preprocessor


import git
os.environ['MFA_ROOT_DOR'] = os.path.join(
    git.Repo(os.getcwd(), search_parent_directories=True).working_dir,
    "MFA"
)

if Define.CUDA_LAUNCH_BLOCKING:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Preprocessor:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.root = args.raw_dir
        self.preprocessed_root = args.preprocessed_dir
        self.raw_parser = get_raw_parser(args.dataset)(Path(args.raw_dir), Path(args.preprocessed_dir))
        self.processor = get_preprocessor(args.dataset)(Path(args.preprocessed_dir))

    def exec(self, force=False):
        self.print_message()
        key_input = ""
        if not force:
            while key_input not in ["y", "Y", "n", "N"]:
                key_input = input("Proceed? ([y/n])? ")
        else:
            key_input = "y"

        if key_input in ["y", "Y"]:
            # 0. Initial features from raw data
            if self.args.parse_raw:
                print("[INFO] Parsing raw corpus...")
                self.raw_parser.parse(n_workers=8)
            # 1. Denoising
            if self.args.denoise:
                print("[INFO] Denoising corpus...")
                torchaudio.set_audio_backend("sox_io")
                self.processor.denoise()
            # 2. Prepare MFA
            if self.args.prepare_mfa:
                print("[INFO] Preparing data for Montreal Force Alignment...")
                self.processor.prepare_mfa(Path(self.preprocessed_root) / "mfa_data")
            # 3. MFA
            if self.args.mfa:
                print("[INFO] Performing Montreal Force Alignment...")
                self.processor.mfa(Path(self.preprocessed_root) / "mfa_data")
            # 4. Create Dataset
            if self.args.preprocess:
                print("[INFO] Preprocess all utterances...")
                self.processor.preprocess()
            if self.args.create_dataset is not None:
                print("[INFO] Creating Training and Validation Dataset...")
                self.processor.split_dataset(self.args.create_dataset)

    def print_message(self):
        print("\n")
        print("------ Preprocessing ------")
        print(f"* Dataset     : {self.dataset}")
        print(f"* Raw Data path   : {self.root}")
        print(f"* Output path : {self.preprocessed_root}")
        print("\n")
        print(" [INFO] The following will be executed:")
        if self.args.parse_raw:
            print("* Parsing raw corpus")
        if self.args.denoise:
            print("* Denoising corpus")
        if self.args.prepare_mfa:
            print("* Preparing data for Montreal Force Alignment")
        if self.args.mfa:
            print("* Montreal Force Alignment")
        if self.args.preprocess:
            print("* Preprocess dataset")
        if self.args.create_dataset is not None:
            print("* Creating Training and Validation Dataset")
        print("\n")


def main(args):
    Define.DEBUG = args.debug
    P = Preprocessor(args)
    P.exec(args.force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("preprocessed_dir", type=str)

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--parse_raw", action="store_true", default=False)
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--prepare_mfa", action="store_true", default=False)
    parser.add_argument("--mfa", action="store_true", default=False)
    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--create_dataset", type=str, help="cleaned data_info path")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
