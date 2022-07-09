import pandas as pd
import re
import sys
import os

total_regex = re.compile(
    r"Applied `(?P<prune_fn>\w+)`."
    r" Pruned: (?P<prev_pruned>\d+)/(?P<prev_total>\d+)"
    r" \((?P<prev_prune_rate>\d+\.\d+)\%\)"
    r" -> (?P<curr_pruned>\d+)/(?P<curr_total>\d+)"
    r" \((?P<curr_prune_rate>\d+\.\d+)\%\)"
)
module_regex = re.compile(
    r"Applied `(?P<prune_fn>\w+)`"
    r" to `(?P<module>\w+\([()\-\w\s=,.]+\)).(?P<param>\w+)`"
    r" with amount=(?P<amount>\d.\d+)."
    r" Pruned: (?P<prev_pruned>\d+)"
    r" \((?P<prev_prune_rate>\d+\.\d+)\%\)"
    r" -> (?P<curr_pruned>\d+)"
    r" \((?P<curr_prune_rate>\d+\.\d+)\%\)"
)

module_df = pd.read_csv(
    "scripts/parameters_to_prune.csv"
)
print(module_df)

print(sys.argv)

data = []
with open(os.path.join(sys.argv[1], "pruning.log"), 'r') as f:
    epoch = -1
    counter = 0
    for line in f:
        result = total_regex.search(line)
        if result is not None:
            epoch += 1
            counter = 0
            prune_dict = result.groupdict()
            for k in prune_dict:
                if k in {"amount"}:
                    prune_dict[k] = float(prune_dict[k])
                if k in {"prev_pruned", "prev_total", "curr_pruned", "curr_total"}:
                    prune_dict[k] = int(prune_dict[k])
                if k in {"prev_prune_rate", "curr_prune_rate"}:
                    prune_dict[k] = float(prune_dict[k]) / 100
            data.append({
                "epoch": epoch,
                "module": "total",
                **prune_dict,
            })
        else:
            result = module_regex.search(line)
            prune_dict = result.groupdict()
            for k in prune_dict:
                if k in {"amount"}:
                    prune_dict[k] = float(prune_dict[k])
                elif k in {"prev_pruned", "prev_total", "curr_pruned", "curr_total"}:
                    prune_dict[k] = int(prune_dict[k])
                elif k in {"prev_prune_rate", "curr_prune_rate"}:
                    prune_dict[k] = float(prune_dict[k]) / 100
                elif k == "module":
                    assert prune_dict[k] == module_df[k][counter-1]
                    prune_dict[k] = module_df["name"][counter-1]
            data.append({
                "epoch": epoch,
                "prev_total": module_df["numel"][counter-1],
                "curr_total": module_df["numel"][counter-1],
                **prune_dict,
            })
        counter += 1

df = pd.DataFrame(data)
print(df)
print(df.columns)
for ep in range(epoch+1):
    csv_file_path = os.path.join(sys.argv[1], "_prune_log", f"epoch={ep}.csv")
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df[df["epoch"].values == ep].to_csv(csv_file_path, index=False)
