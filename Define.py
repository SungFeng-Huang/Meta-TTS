import numpy as np
import json


def merge_stats(stats_dict, keys):
    num = len(keys)
    pmi, pmx, pmu, pstd, emi, emx, emu, estd = \
        np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0, np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0
    for k in keys:
        pmu += stats_dict[k][2]
        pstd += stats_dict[k][3] ** 2
        emu += stats_dict[k][6]
        estd += stats_dict[k][7] ** 2
        pmi = min(pmi, stats_dict[k][0] * stats_dict[k][3] + stats_dict[k][2])
        pmx = max(pmx, stats_dict[k][1] * stats_dict[k][3] + stats_dict[k][2])
        emi = min(emi, stats_dict[k][4] * stats_dict[k][7] + stats_dict[k][6])
        emx = max(emx, stats_dict[k][5] * stats_dict[k][7] + stats_dict[k][6])

    pmu, pstd, emu, estd = pmu / num, (pstd / num) ** 0.5, emu / num, (estd / num) ** 0.5
    pmi, pmx, emi, emx = (pmi - pmu) / pstd, (pmx - pmu) / pstd, (emi - emu) / estd, (emx - emu) / estd
    
    return [pmi, pmx, pmu, pstd, emi, emx, emu, estd]


STATSDICT = {
    0: "./preprocessed_data/LibriTTS",
    1: "./preprocessed_data/AISHELL-3",
    2: "./preprocessed_data/GlobalPhone/fr",
    3: "./preprocessed_data/CSS10/german",
    4: "./preprocessed_data/GlobalPhone/es",
    5: "./preprocessed_data/GlobalPhone/es",
    6: "./preprocessed_data/JVS",
    7: "./preprocessed_data/GlobalPhone/cz",
    8: "./preprocessed_data/kss",
}

ALLSTATS = {}
for k, v in STATSDICT.items():
    try:
        with open(f"{v}/stats.json") as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"]
            ALLSTATS[k] = stats
    except:
        ALLSTATS[k] = None

LANGUSAGE = [0, 1, 2, 5, 7, 8]  # Dirty! Need to adjust manually for every experiment.
ALLSTATS["global"] = merge_stats(ALLSTATS, LANGUSAGE)


if __name__ == "__main__":
    import json
    print(json.dumps(ALLSTATS, indent=4))
