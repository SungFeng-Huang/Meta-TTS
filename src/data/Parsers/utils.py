import os

from .parser import DataParser


def read_queries_from_txt(path):
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == '\n':
                continue
            n, s, t, r = line.strip("\n").split("|")
            res.append({
                "basename": n,
                "spk": s,
            })
    return res


def write_queries_to_txt(data_parser: DataParser, queries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_parser.phoneme.read_all()
    data_parser.text.read_all()
    lines = []
    for query in queries:
        try:
            line = [query["basename"], query["spk"]]
            line.append(f"{{{data_parser.phoneme.read_from_query(query)}}}")
            line.append(data_parser.text.read_from_query(query))
            lines.append(line)
        except:
            print("Please delete phoneme cache and text cache and try again.")
            print("If not working, phoneme feature/text feature does not contain such query.")
            print("Failed: ", query)
            raise
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write("|".join(line))
            f.write('\n')
