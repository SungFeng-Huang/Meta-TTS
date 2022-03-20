import argparse
import shutil
import yaml
import os
from copy import deepcopy
from tqdm import tqdm

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    subsets = [s for dset in config["subsets"] for s in config["subsets"][dset]]
    for subset in tqdm(subsets):
        subconfig = deepcopy(config)
        subconfig["path"]["raw_path"] = os.path.join(
            config["path"]["raw_path"], subset
        )
        subconfig["path"]["preprocessed_path"] = os.path.join(
            config["path"]["preprocessed_path"], subset
        )
        os.makedirs(subconfig["path"]["preprocessed_path"], exist_ok=True)
        if (not os.path.exists(os.path.join(subconfig["path"]["preprocessed_path"], "TextGrid"))
                and os.path.exists(os.path.join(config["path"]["preprocessed_path"], "TextGrid"))):
            os.symlink(
                os.path.join('/'.join([".."] * len(subset.split('/'))), "TextGrid"),
                os.path.join(subconfig["path"]["preprocessed_path"], "TextGrid")
            )
        tqdm.write(subset)
        preprocessor = Preprocessor(subconfig)
        preprocessor.build_from_path()

    # if "LibriTTS" in config["dataset"]:
        # if config["subsets"]["train"] == "train-clean":
            # config["subsets"]["train"] = ["train-clean-100", "train-clean-360"]
        # elif config["subsets"]["train"] == "train-all":
            # config["subsets"]["train"] = ["train-clean-100", "train-clean-360", "train-other-500"]
    # preprocessor = Preprocessor(config)
    # preprocessor.build_from_path()
    # if "LibriTTS" in config["dataset"]:
        # if config["subsets"]["train"] == ["train-clean-100", "train-clean-360"]:
            # with open(os.path.join(config["path"]["preprocessed_path"], "train-clean.txt"), 'wb') as out_file:
                # with open(os.path.join(config["path"]["preprocessed_path"], "train-clean-100.txt"), 'rb') as in_file:
                    # shutil.copyfileobj(in_file, out_file)
                # with open(os.path.join(config["path"]["preprocessed_path"], "train-clean-360.txt"), 'rb') as in_file:
                    # shutil.copyfileobj(in_file, out_file)
        # elif config["subsets"]["train"] == ["train-clean-100", "train-clean-360", "train-other-500"]:
            # with open(os.path.join(config["path"]["preprocessed_path"], "train-all.txt"), 'wb') as out_file:
                # with open(os.path.join(config["path"]["preprocessed_path"], "train-clean-100.txt"), 'rb') as in_file:
                    # shutil.copyfileobj(in_file, out_file)
                # with open(os.path.join(config["path"]["preprocessed_path"], "train-clean-360.txt"), 'rb') as in_file:
                    # shutil.copyfileobj(in_file, out_file)
                # with open(os.path.join(config["path"]["preprocessed_path"], "train-other-500.txt"), 'rb') as in_file:
                    # shutil.copyfileobj(in_file, out_file)
