from collections import defaultdict
import seutil as su
from jsonargparse import CLI
from typing import List, Dict, Any
from statistics import mean, median
from tqdm import tqdm
import difflib

from etestgen.macros import Macros
from etestgen.data.data import TData, parse_data_cls
from etestgen.data.utils import load_dataset, save_dataset

logger = su.log.get_logger(__name__, su.log.INFO)


class MetricsCollector:

    SPLITS = ["train", "valid", "test"]

    def __init__(self) -> None:
        self.metrics_dir = Macros.results_dir / "stats"
        su.io.mkdir(self.metrics_dir)

    def collect_stmt_data_stats(self, split_set: str, data_cls: str="MUT2E", exp: str = "mut2e-stmt"):
        
        data_cls = parse_data_cls(data_cls)
        data_dir = Macros.work_dir / "setup" / exp / split_set
        gold_stmts = su.io.load(data_dir / "gold_stmts.jsonl")
        with tqdm("Loading data") as pbar:
            dataset: List[data_cls] = load_dataset(
                data_dir, clz=data_cls, pbar=pbar
            )
        
        projects_set = set([d.project for d in dataset])
        etests_set = set([d.id for d in dataset])
        stmts_toks_num = [len(gold_st) for gold_st in gold_stmts]

        metrics = {
            "num-projects": len(projects_set),
            "num-estests": len(etests_set),
            "num-stmts": len(gold_stmts),
            "mean-stmts-toks": mean(stmts_toks_num),
        }
        su.io.dump(self.metrics_dir / f"metrics-{exp}-{split_set.replace('/', '-')}-data-stats.json", metrics, su.io.Fmt.jsonPretty)


if __name__ == "__main__":
    CLI(MetricsCollector, as_positional=False)
