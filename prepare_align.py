import os
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

from preprocessor import libritts, vctk


def main(config):
    if "LibriTTS" in config["dataset"]:
        corpus_path = config["path"]["corpus_path"]
        raw_path = config["path"]["raw_path"]
        dsets = []
        for dmode, dset in config["subsets"].items():
            if dset == "train-clean":
                dsets += ["train-clean-100", "train-clean-360"]
            elif dset == "train-all":
                dsets += ["train-clean-100", "train-clean-360", "train-other-500"]
            # if isinstance(dset, list):
                # dsets += dset
            # elif isinstance(dset, str):
                # dsets.append(dset)
        for dset in dsets:
            config["path"]["corpus_path"] = os.path.join(corpus_path, dset)
            config["path"]["raw_path"] = os.path.join(raw_path, dset)
            libritts.prepare_align(config)
    if "VCTK" in config["dataset"]:
        corpus_path = config["path"]["corpus_path"]
        raw_path = config["path"]["raw_path"]
        df = pd.read_csv(os.path.join(corpus_path, "speaker-info.csv"))
        df = df.fillna("Unknown")
        for accent in tqdm(df.groupby(["ACCENTS", "REGION"])):
            subset = '/'.join(accent[0])
            speakers = accent[1]["pID"].values.tolist()
        # for dmode, dset in config["subsets"].items():
            # # config["path"]["corpus_path"] = corpus_path
            config["path"]["raw_path"] = os.path.join(raw_path, subset)
            if not os.path.exists(config["path"]["raw_path"]):
                tqdm.write(subset)
                vctk.prepare_align(config, speakers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
