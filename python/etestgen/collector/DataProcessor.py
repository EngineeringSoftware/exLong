from typing import List
import seutil as su
from seutil.project import Project
from seutil.maven import MavenModule, MavenProject
from tqdm import tqdm
from jsonargparse import CLI
from collections import defaultdict
import numpy as np
import random

random.seed(42)

from etestgen.data.tool import Tool
from etestgen.macros import Macros
from etestgen.eval.analyze_throws_coverage import ThrowsCoverageAnalyzer, Coverage
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.data.data import TData, parse_data_cls
from etestgen.data.data import DataNE2E

logger = su.log.get_logger(__name__)


class DataProcessor:
    def __init__(self, config_file: str):
        self.config = su.io.load(config_file)
        self.data_dir = Macros.work_dir / "setup" / self.config["setup"]
        su.io.mkdir(self.data_dir)

    def process_ins2e_dataset(self):
        """
        The dataset contain instructions from GPT3.5 but only for test set
        """
        eval_set = "eval/test"

        # load raw dataset
        gpt_outputs = su.io.load(
            Macros.exp_dir / "stack2inst" / "gpt-3.5-turbo-0613" / "model-outputs.jsonl"
        )
        condition_instructions = [out["responses"][0] for out in gpt_outputs]
        su.io.dump(
            self.data_dir / eval_set / "instruction.jsonl", condition_instructions
        )

    def write_subset_dataset(self, selected_ids: List):

        # load raw dataset
        data = load_dataset(Macros.data_dir / "ne2e-test", clz=DataNE2E)
        subset_data = [dt for dt in data if dt.id in selected_ids]
        save_dataset(Macros.data_dir / "rq1-eval", subset_data)


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.INFO)
    CLI(DataProcessor, as_positional=False)
