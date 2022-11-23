"""
Most extreme task contains m support + 64 query with phoneme coverage condition.
After that, we extend to n support base on m support so that we can use same query set.
Since extension in support set will not break phoneme coverage condition.
"""
import os
import random
from typing import List, Dict
from tqdm import tqdm
import yaml

from src.data.Parsers.utils import read_queries_from_txt, write_queries_to_txt
from src.data.Parsers.parser import DataParser


def collect_phonemes(data_parser: DataParser, queries):
    res = set()
    for query in queries:
        phns = data_parser.phoneme.read_from_query(query).split()
        res.update(phns)
    return res


class TaskGenerator(object):
    def __init__(self, dataset_name: str, preprocessed_dir: str, lang_id, max_trial=1000) -> None:
        self.data_parser = DataParser(preprocessed_dir)
        self.dataset_name = dataset_name
        self.lang_id = lang_id
        self.max_trial = max_trial

    def generate_base_sup_candidates(self, queries, n_sup: int, n_candidates: int):
        print("Generate base support set candidates...")
        res = []
        for _ in tqdm(range(n_candidates)):
            cand = random.sample(queries, n_sup)
            phns = collect_phonemes(self.data_parser, cand)
            res.append((phns, cand))
        res.sort(key=lambda x: len(x[0]), reverse=True)
        return res

    def generate_base_tasks(self, queries, n_sup: int, n_qry: int, n_tasks: int):
        res = []
        candidates = self.generate_base_sup_candidates(queries, n_sup, n_candidates=4000)
        for (phns, sup) in candidates:
            # remove sup from queries
            sup_names = [q["basename"] for q in sup]
            pool = [q for q in queries if q["basename"] not in sup_names]

            # try to generate query set
            fail_cnt = 0
            qry = []
            while fail_cnt < self.max_trial and len(qry) < n_qry and pool:
                idx = random.randint(0, len(pool) - 1)
                q = pool[idx]
                if phns >= collect_phonemes(self.data_parser, [q]):
                    qry.append(q)
                else:
                    fail_cnt += 1
                pool.pop(idx)

            if len(qry) == n_qry:
                res.append((sup, qry))
                print(f"Find! ({len(res)}/{n_tasks}), support set contains {len(phns)} phonemes.")
            if len(res) == n_tasks:
                return res
        raise ValueError("Fail to generate support set for most extreme case...")  

    def generate_extend_tasks(self, queries, shots: List[int], base_task):
        sup, qry = base_task
        assert min(shots) == len(sup)
        assert collect_phonemes(self.data_parser, sup) >= collect_phonemes(self.data_parser, qry)

        # remove sup/qry from queries
        names = [q["basename"] for q in sup + qry]
        pool = [q for q in queries if q["basename"] not in names]

        res = [base_task]
        for n in shots[1:]:
            sup_ext = random.sample(pool, n - len(sup))
            res.append((sup + sup_ext, qry))

        return res

    def generate(
        self,
        src_txt_path, output_dir,
        shots: List[int], n_qry: int=64, n_tasks: int=20
    ):  
        os.makedirs(output_dir, exist_ok=True)
        queries = read_queries_from_txt(src_txt_path)

        print("Generate base support/query sets...")
        base_tasks = self.generate_base_tasks(queries, min(shots), n_qry, n_tasks)

        print("Extend base support/query sets...")
        for i, base_task in tqdm(enumerate(base_tasks)):
            extended_tasks = self.generate_extend_tasks(queries, shots, base_task)
            for n_sup, task in zip(shots, extended_tasks):
                sup, qry = task
                dst = f"{output_dir}/{n_sup}-shot/task-{i}"
                write_queries_to_txt(self.data_parser, sup, f"{dst}/train.txt")
                write_queries_to_txt(self.data_parser, qry, f"{dst}/val.txt")
                with open(f"{dst}/config.yaml", 'w') as f:
                    f.write(yaml.dump(self.config_template(), sort_keys=False))

    def config_template(self) -> Dict:
        template = {
            "dataset": self.dataset_name,
            "lang_id": self.lang_id,
            "data_dir": self.data_parser.root,
            "subsets": {
                "train": "train.txt",
                "val": "val.txt",
                "test": "val.txt"
            }
        }
        return template
            

if __name__ == "__main__":
    random.seed(666)
    generator = TaskGenerator("kss", "preprocessed_data/kss", lang_id="ko")
    generator.generate(
        src_txt_path="_data/kss/val.txt",
        output_dir="_data/kss/few-shot",
        shots=[4, 8, 16, 32, 64, 128],
        n_qry=64,
        n_tasks=20
    )

    generator = TaskGenerator("JSUT", "preprocessed_data/JSUT", lang_id="jp")
    generator.generate(
        src_txt_path="_data/JSUT/val.txt",
        output_dir="_data/JSUT/few-shot",
        shots=[4, 8, 16, 32, 64, 128],
        n_qry=64,
        n_tasks=20
    )

    generator = TaskGenerator("CSS10-german", "preprocessed_data/CSS10/german", lang_id="de")
    generator.generate(
        src_txt_path="_data/CSS10/german/val.txt",
        output_dir="_data/CSS10/german/few-shot",
        shots=[4, 8, 16, 32, 64, 128],
        n_qry=64,
        n_tasks=20
    )

    generator = TaskGenerator("CSS10-spanish", "preprocessed_data/CSS10/spanish", lang_id="es")
    generator.generate(
        src_txt_path="_data/CSS10/spanish/val.txt",
        output_dir="_data/CSS10/spanish/few-shot",
        shots=[4, 8, 16, 32, 64, 128],
        n_qry=64,
        n_tasks=20
    )
