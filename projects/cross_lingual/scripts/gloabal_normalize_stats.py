import numpy as np
import json

from src.data.Parsers.parser import DataParser


def merge_stats(stats_dict, keys):
    num = len(keys)
    pmi, pmx, pmu, pstd, emi, emx, emu, estd = \
        np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0, np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0
    for k in keys:
        pmu += stats_dict[k][2]
        pstd += stats_dict[k][3] ** 2
        emu += stats_dict[k][6]
        estd += stats_dict[k][7] ** 2
        pmi = min(pmi, stats_dict[k][0])
        pmx = max(pmx, stats_dict[k][1])
        emi = min(emi, stats_dict[k][4])
        emx = max(emx, stats_dict[k][5])

    pmu, pstd, emu, estd = pmu / num, (pstd / num) ** 0.5, emu / num, (estd / num) ** 0.5
    
    return [pmi, pmx, pmu, pstd, emi, emx, emu, estd]


if __name__ == "__main__":
    datasets = [
        "preprocessed_data/LJSpeech",
        "preprocessed_data/LibriTTS",
        "preprocessed_data/AISHELL-3",
        "preprocessed_data/JSUT",
        "preprocessed_data/kss",
        "preprocessed_data/CSS10/german",
        "preprocessed_data/CSS10/french",
        "preprocessed_data/CSS10/spanish",
    ]
    all_stats = {}
    for d in datasets:
        data_parser = DataParser(d)
        with open(data_parser.stats_path) as f:
            stats = json.load(f)
        stats = stats["pitch"] + stats["energy"]
        all_stats[d] = stats
    
    res = merge_stats(all_stats, list(all_stats.keys()))
    stats = {
        "pitch": res[:4],
        "energy": res[4:]
    }
    with open("stats.json", 'w', encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
