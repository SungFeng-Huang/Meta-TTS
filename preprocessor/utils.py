import os
import json
import yaml
import shutil
import argparse
import numpy as np
from tqdm import tqdm


def merge_stats(*all_stats):
    if len(all_stats) == 0:
        return {}
    elif len(all_stats) == 1:
        return all_stats[0]

    origin_stats, other_stats = all_stats[0], all_stats[1:]
    new_stats = origin_stats
    for stats in other_stats:
        old_stats = new_stats
        new_stats = {}
        for k in ["pitch", "energy"]:
            M = old_stats[k]["n_samples"]
            N = stats[k]["n_samples"]
            mu_m = old_stats[k]["mean"]
            mu_n = stats[k]["mean"]
            var_m = old_stats[k]["std"] ** 2
            var_n = stats[k]["std"] ** 2
            new_stats[k] = {
                "n_samples": M + N,
                "min": min(old_stats[k]["min"], stats[k]["min"]),
                "max": max(old_stats[k]["max"], stats[k]["max"]),
                "mean": (M*mu_m + N*mu_n) / (M + N),
                "std": np.sqrt(
                    (M*var_m + N*var_n + M*N/(M+N)*((mu_m-mu_n)**2)) / (M + N)
                ),
            }
    return new_stats


def merge_speaker_map(*speaker_maps):
    new_speaker_map = {}
    i = 0
    for spk_map in speaker_maps:
        for spk in spk_map:
            new_speaker_map[spk] = i
            i += 1
    return new_speaker_map


def merge_subsets(config):
    root_path = config["root"]["preprocessed_path"]
    new_path = os.path.join(config["root"]["preprocessed_path"], config["path"]["child_path"])
    level = len(config["path"]["child_path"].split('/'))

    os.makedirs((os.path.join(new_path, "mel")), exist_ok=True)
    os.makedirs((os.path.join(new_path, "pitch")), exist_ok=True)
    os.makedirs((os.path.join(new_path, "energy")), exist_ok=True)
    os.makedirs((os.path.join(new_path, "duration")), exist_ok=True)
    os.makedirs((os.path.join(new_path, "spk_ref_mel_slices")), exist_ok=True)

    all_speaker_maps = []
    all_stats = []
    for dset in tqdm(config["subsets"]):
        for subset in tqdm(config["subsets"][dset], desc=dset, leave=False):
            subset_path = os.path.join(root_path, subset)
            assert os.path.exists(subset_path)
            for dirname in tqdm(["mel", "pitch", "energy", "duration", "spk_ref_mel_slices"],
                                desc=subset, leave=False):
                for filename in tqdm(os.listdir(os.path.join(subset_path, dirname)),
                                     desc=dirname, leave=False):
                    if not os.path.exists(os.path.join(new_path, dirname, filename)):
                        os.symlink(
                            os.path.join("/".join([".."] * (level+1)),
                                         subset, dirname, filename),
                            os.path.join(new_path, dirname, filename)
                        )
                    else:
                        # tqdm.write(f"File {os.path.join(new_path, dirname, filename)} exists...")
                        # continue
                        pass

            subset_level = len(subset.split('/')) - 1
            if not os.path.exists(os.path.join(new_path, f"{subset}.txt")):
                if subset_level > 0:
                    # tqdm.write(os.path.join(new_path, *subset.split('/')[:-1]))
                    os.makedirs(os.path.join(new_path, *subset.split('/')[:-1]), exist_ok=True)
                # tqdm.write(os.path.join("/".join([".."] * (level + subset_level)), subset, "total.txt"))
                # tqdm.write(os.path.join(new_path, f"{subset}.txt"))
                os.symlink(
                    os.path.join("/".join([".."] * (level + subset_level)), subset, "total.txt"),
                    os.path.join(new_path, f"{subset}.txt")
                )
            else:
                # target_filename = os.path.join(new_path, f"{subset}.txt")
                # tqdm.write(f"File {target_filename} exists...")
                # continue
                pass

            with open(os.path.join(subset_path, "speakers.json"), 'r') as f:
                all_speaker_maps.append(json.load(f))
            with open(os.path.join(subset_path, "stats.json"), 'r') as f:
                all_stats.append(json.load(f))

    with open(os.path.join(new_path, "total.txt"), 'wb') as total_file:
        for dset in config["subsets"]:
            if len(config["subsets"][dset]) == 0:
                continue
            with open(os.path.join(new_path, f"{dset}.txt"), 'wb') as out_file:
                for subset in config["subsets"][dset]:
                    with open(os.path.join(new_path, f"{subset}.txt"), 'rb') as in_file:
                        shutil.copyfileobj(in_file, out_file)
                    with open(os.path.join(new_path, f"{subset}.txt"), 'rb') as in_file:
                        shutil.copyfileobj(in_file, total_file)

    speaker_map = merge_speaker_map(*all_speaker_maps)
    with open(os.path.join(new_path, "speakers.json"), 'w') as f:
        f.write(json.dumps(speaker_map, indent=4))

    stats = merge_stats(*all_stats)
    with open(os.path.join(new_path, "stats.json"), 'w') as f:
        f.write(json.dumps(stats, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    merge_subsets(config)
        # subconfig = deepcopy(config)
        # subconfig["path"]["raw_path"] = os.path.join(
            # config["path"]["raw_path"], dset
        # )
        # subconfig["path"]["preprocessed_path"] = os.path.join(
            # config["path"]["preprocessed_path"], dset
        # )
        # preprocessor = Preprocessor(config)
        # preprocessor.build_from_path()
