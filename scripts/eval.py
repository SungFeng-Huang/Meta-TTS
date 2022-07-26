import pandas as pd
import os
import glob
import subprocess
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

root = "new_output/prune_accent"
# dir/sanity_check/csv/validate/epoch=0-end.csv
# dir/train/csv/{support,query}/epoch=*.csv
# dir/validate/csv/validate/epoch=*-{start,end}.csv
def get_dir_ver_dict(accent, speaker, prune):
    output = {}
    for dir_name in sorted(glob.glob(
            os.path.join(f"{root}/{accent}/{speaker}/prune={prune}",
                        "lightning_logs/version_*"))):
        version_name = os.path.basename(dir_name)
        output[version_name] = dir_name
        if len(output) == 5:
            break
    return output

def get_train_df(accent, speaker, prune):
    def get_sup_epoch_dict(dir):
        output = {}
        for dir_name in sorted(glob.glob(
                f"{dir}/fit/train/csv/support/epoch=*.csv")):
            epoch = int(os.path.basename(dir_name)[6:-4])
            output[epoch] = dir_name
        return output

    def get_qry_epoch_dict(dir):
        output = {}
        for dir_name in sorted(glob.glob(
                f"{dir}/fit/train/csv/query/epoch=*.csv")):
            epoch = int(os.path.basename(dir_name)[6:-4])
            output[epoch] = dir_name
        return output

    dfs = []
    type_fn = {
        "support": get_sup_epoch_dict,
        "query": get_qry_epoch_dict,
    }
    metrics = [f"{s}_{t}_{m}"
               for s in {"sup", "qry"}
               for t in {"accent", "speaker"}
               for m in {"prob", "acc"}]
    for ver, ver_dir in get_dir_ver_dict(accent, speaker, prune).items():
        for _type in type_fn:
            for epoch, _csv in type_fn[_type](ver_dir).items():
                df = pd.read_csv(_csv)
                assert "batch_idx" in df
                df_dict = {
                    "epoch": epoch if prune == 0.1 else epoch*2,
                    "version": ver,
                    "accent": accent,
                    "speaker": speaker,
                    "prune_rate": prune,
                    "_type": _type,
                    "batch_idx": df["batch_idx"].max(),
                }
                dfs.append(df_dict)
    if len(dfs) > 0:
        df = pd.DataFrame(dfs)
        df.apply(pd.to_numeric, errors='ignore')
        return df
    return None

def get_val_df(accent, speaker, prune):
    def get_val_start_epoch_dict(dir):
        output = {}
        for dir_name in sorted(glob.glob(
                f"{dir}/fit/validate/csv/validate/epoch=*-start.csv")):
            epoch = int(os.path.basename(dir_name)[6:-10])
            output[epoch] = dir_name
        return output

    def get_val_end_epoch_dict(dir):
        output = {}
        for dir_name in sorted(glob.glob(
                f"{dir}/fit/validate/csv/validate/epoch=*-end.csv")):
            epoch = int(os.path.basename(dir_name)[6:-8])
            output[epoch] = dir_name
        return output

    dfs = []
    type_fn = {
        "start": get_val_start_epoch_dict,
        "end": get_val_end_epoch_dict,
    }
    metrics = [f"val_{t}_{m}"
               for t in {"accent", "speaker"}
               for m in {"prob", "acc"}]
    for ver, ver_dir in get_dir_ver_dict(accent, speaker, prune).items():
        for _type in type_fn:
            for epoch, val_start_csv in type_fn[_type](ver_dir).items():
                df = pd.read_csv(val_start_csv)
                assert "epoch" in df
                if prune == 0.2:
                    df["epoch"] = df["epoch"].multiply(2)
                df["version"] = ver
                df["accent"] = accent
                df["speaker"] = speaker
                df["prune_rate"] = prune
                df["_time"] = _type
                df.apply(pd.to_numeric, errors='ignore')
                # print(df)
                df = df.melt(
                    id_vars=[k for k in df.keys() if k not in metrics],
                    value_vars=metrics,
                    var_name="metric_name", value_name="metric_value",
                )
                dfs.append(df)
    if len(dfs) > 0:
        df = pd.concat(dfs, ignore_index=True)
        return df
    return None

def get_prune_df(accent, speaker, prune):
    def get_prune_log_epoch_dict(dir):
        output = {}
        for dir_name in sorted(glob.glob(
                f"{dir}/fit/prune_log/epoch=*.csv")):
            epoch = int(os.path.basename(dir_name)[6:-4])
            output[epoch] = dir_name
        return output

    dfs = []
    for ver, ver_dir in get_dir_ver_dict(accent, speaker, prune).items():
        for epoch, prune_log_csv in get_prune_log_epoch_dict(ver_dir).items():
            df = pd.read_csv(prune_log_csv)
            sub_dicts = []
            for module in {"encoder", "variance_adaptor", "decoder",
                           "mel_linear", "postnet", "speaker_emb"}:
                sub_df = df[df["module"].str.startswith(module, na=False)]
                sub_dict = {
                    "epoch": epoch if prune == 0.1 else epoch*2,
                    "module": module,
                    "version": ver,
                    "prev_pruned": sub_df["prev_pruned"].sum(),
                    "prev_total": sub_df["prev_total"].sum(),
                    "curr_pruned": sub_df["curr_pruned"].sum(),
                    "curr_total": sub_df["curr_total"].sum(),
                }
                sub_dict["prev_prune_rate"] = \
                    float(sub_dict["prev_pruned"]) / sub_dict["prev_total"]
                sub_dict["curr_prune_rate"] = \
                    float(sub_dict["curr_pruned"]) / sub_dict["curr_total"]
                sub_dicts.append(sub_dict)
            dfs.append(pd.DataFrame(sub_dicts))
    if len(dfs) > 0:
        df = pd.concat(dfs, ignore_index=True)
        df["accent"] = accent
        df["speaker"] = speaker
        df["prune_rate"] = prune
        return df
    return None

def plot_train_batch_num(df, accents, _type="support", col="accent"):
    if _type is not None:
        df = df[df["_type"].values == _type]
    with sns.axes_style("whitegrid"):
        g = sns.relplot(
            kind="line",
            data=df,
            x="epoch",
            y="batch_idx",
            # hue="metric_name",
            style="prune_rate",
            legend="full",
            err_style="bars",
            err_kws={"capsize": 3.},
            ci="sd",
            col=col,
            col_wrap=4,
        )

def plot_val(df, accents, _type=None, col="accent"):
    if _type is not None:
        df = df[df["_time"].values == _type]
    with sns.axes_style("whitegrid"):
        for col_order in (accents[:4], accents[4:-4], accents[-4:]):
            g = sns.relplot(
                kind="line",
                data=df,
                x="epoch",
                y="metric_value",
                hue="metric_name",
                style="prune_rate",
                legend="full",
                err_style="bars",
                err_kws={"capsize": 3.},
                ci="sd",
                row="_time",
                col=col,
                col_order=col_order,
            )
        # plt.legend(loc='lower center')
        # plt.legend(bbox_to_anchor=(0.5, 0.05), loc='upper center', borderaxespad=0)
        # leg = g._legend
        # leg.set_bbox_to_anchor([0.5, -0.05])  # coordinates of lower left of bounding box
        # leg._loc = 9

def plot_prune(df, accents, _type=None, col="accent"):
    if _type is not None:
        df = df[df["module"].values == _type]
    with sns.axes_style("whitegrid"):
        sns.relplot(
            kind="line",
            data=df,
            x="epoch",
            y="prev_prune_rate",
            hue="module",
            style="prune_rate",
            legend="full",
            err_style="bars",
            err_kws={"capsize": 3.},
            ci="sd",
            col=col,
            col_wrap=4,
        )

def plot_all_prune(prunes=None):
    vctk_df = pd.read_csv("preprocessed_data/VCTK-speaker-info.csv")
    dfs = []
    accents = set()
    if prunes is None:
        prunes = {0.1, 0.2}
    for prune in prunes:
        for _, row in vctk_df.iterrows():
            accent = row["ACCENTS"]
            speaker = row["pID"]
            print(accent, speaker, prune)
            if not os.path.exists(
                    f"{root}/{accent}/{speaker}/prune={prune}"):
                continue
            accents.add(accent)
            df = get_prune_df(accent, speaker, prune)
            if df is not None:
                print(accent, speaker, prune)
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    accents = sorted(accents)
    plot_prune(df, accents)

def plot_all_train_batch_num(prunes=None):
    vctk_df = pd.read_csv("preprocessed_data/VCTK-speaker-info.csv")
    dfs = []
    accents = set()
    if prunes is None:
        prunes = {0.1, 0.2}
    for prune in prunes:
        for _, row in vctk_df.iterrows():
            accent = row["ACCENTS"]
            speaker = row["pID"]
            print(accent, speaker, prune)
            if not os.path.exists(
                    f"{root}/{accent}/{speaker}/prune={prune}"):
                continue
            accents.add(accent)
            df = get_train_df(accent, speaker, prune)
            if df is not None:
                print(accent, speaker, prune)
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    accents = sorted(accents)
    plot_train_batch_num(df, accents)

def plot_all_val(prunes=None):
    vctk_df = pd.read_csv("preprocessed_data/VCTK-speaker-info.csv")
    dfs = []
    accents = set()
    if prunes is None:
        prunes = {0.1, 0.2}
    for prune in prunes:
        for _, row in vctk_df.iterrows():
            accent = row["ACCENTS"]
            speaker = row["pID"]
            print(accent, speaker, prune)
            if not os.path.exists(
                    f"{root}/{accent}/{speaker}/prune={prune}"):
                continue
            accents.add(accent)
            df = get_val_df(accent, speaker, prune)
            if df is not None:
                print(accent, speaker, prune)
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    accents = sorted(accents)
    plot_val(df, accents=accents)

if __name__ == "__main__":
    plot_all_train_batch_num()
    # plot_all_val()
    plot_all_prune()
    plt.show()
