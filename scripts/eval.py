import pandas as pd
import os
import glob
import subprocess
from tqdm import tqdm

root = "output/prune_accent"
# accent = "English"
# speaker = "p225"
# prune = 0.1
# dir/sanity_check/csv/validate/epoch=0-end.csv
# dir/train/csv/{support,validate}/epoch=*.csv
# dir/validate/csv/validate/epoch=*-{start,end}.csv
def get_dirs(accent, speaker, prune):
    return glob.glob(f"{root}/{accent}/{speaker}/prune={prune}/lightning_logs/version_*")

def val_start_path(dir, epoch):
    return f"{dir}/fit/validate/csv/validate/epoch={epoch}-start.csv"

def val_end_path(dir, epoch):
    return f"{dir}/fit/validate/csv/validate/epoch={epoch}-end.csv"

vctk_df = pd.read_csv("preprocessed_data/VCTK-speaker-info.csv")
for _, row in vctk_df.iterrows():
    if not os.path.exists(f"output/prune_accent/{row['ACCENTS']}/{row['pID']}"):
        continue
    print(row["pID"], row["ACCENTS"])
    val_start_dfs = [pd.read_csv(val_start_path(dir, epoch))
                     for dir in get_dirs(row["ACCENTS"], row["pID"], 0.1)
                     for epoch in range(10)]
    for df in val_start_dfs:
        print(df)
print(vctk_df)

# with open("scripts/unparsed.txt", 'w') as f:
#     for accent_dir in tqdm(glob.glob(f"output/prune_accent/*"), leave=False):
#         accent = accent_dir.rsplit('/', 1)[1]
#         for speaker_dir in tqdm(glob.glob(f"{accent_dir}/*"), leave=False):
#             speaker = speaker_dir.rsplit('/', 1)[1]
#             for prune_dir in tqdm(glob.glob(f"{speaker_dir}/prune=*"), leave=False):
#                 prune = float(prune_dir.rsplit('/', 1)[1][6:])
#                 for dir in tqdm(glob.glob(f"{prune_dir}/lightning_logs/*/fit/"),
#                                 leave=False):
#                     n_lines = int(subprocess.check_output(
#                         ['wc', '-l', os.path.join(dir, "pruning.log")]).split()[0])
#                     if n_lines < 2170:
#                         f.write(str(dir)+'\n')
#                         f.write(str(os.listdir(dir))+'\n')
#                         f.write(str(n_lines)+'\n')
#                     elif "_prune_log" in os.listdir(dir):
#                         tqdm.write(str(dir))
#                         tqdm.write(str(os.listdir(dir)))
#                         tqdm.write(str(os.listdir(os.path.join(dir, "_prune_log")))+'\n')
