import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
import collections
import copy
import seutil as su
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw
from tqdm import tqdm

from etestgen.data.data import TData, parse_data_cls
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.macros import Macros

logger = su.log.get_logger(__name__)


class EvalSetup:
    def __init__(self, data_cls: str = "MUT2E"):
        self.config = dict({k: v for k, v in locals().items() if k != "self"})
        self.data_cls = parse_data_cls(data_cls)

    def prepare_train(
        self,
        data_dir: Path_drw,
        split_dir: Path_drw,
        out_dir: Path_dc,
    ):
        """
        Prepares the train and validation sets under the requested setup.
        :param data_dir: source data directory.
        :param split_dir: split ids directory.
        :param out_dir: output data directory, which will contain train/ and val/ datasets.
        """
        train_config = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "data_dir", "split_dir", "out_dir"]
        }
        self.config.update(train_config)
        logger.info(f"Config:\n{self.config}")

        data_dir = Path(data_dir.abs_path)
        split_dir = Path(split_dir.abs_path)
        out_dir = Path(out_dir.abs_path)

        su.io.mkdir(out_dir, fresh=True)
        su.io.dump(out_dir / "config.json", self.config, su.io.Fmt.jsonNoSort)

        # Load ids
        sn2ids: Dict[str, List[str]] = {}
        for sn in [Macros.train, Macros.val]:
            sn2ids[sn] = su.io.load(split_dir / f"{sn}.json")
        all_ids: Set[str] = set(sum(sn2ids.values(), []))
        # Load data
        with tqdm("Loading data") as pbar:
            dataset: List[TData] = load_dataset(
                data_dir, clz=self.data_cls, expected_ids=all_ids, pbar=pbar
            )
        indexed_dataset: Dict[int, TData] = {d.id: d for d in dataset}

        sn2ds = {sn: [indexed_dataset[i] for i in ids] for sn, ids in sn2ids.items()}

        # Save data
        for sn in [Macros.train, Macros.val]:
            su.io.mkdir(out_dir / sn)
            save_dataset(out_dir / sn, sn2ds[sn])

    def prepare_eval(
        self,
        data_dir: Path_drw,
        split_dir: Path_drw,
        out_dir: Path_dc,
    ):
        """
        Prepares the evaluation (val & test) sets under the requested setup.

        :param data_dir: source data directory.
        :param split_dir: split ids directory.
        :param out_dir: output data directory, which will contain val/ and test/ datasets.
        :param seed: random seed (for sampling data).
        """
        eval_config = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "data_dir", "split_dir", "out_dir"]
        }
        self.config.update(eval_config)
        logger.info(f"Config:\n{self.config}")

        data_dir = Path(data_dir.abs_path)
        split_dir = Path(split_dir.abs_path)
        out_dir = Path(out_dir.abs_path)

        su.io.mkdir(out_dir, fresh=True)
        su.io.dump(out_dir / "config.json", self.config, su.io.Fmt.jsonNoSort)

        # Load data ids
        sn2ids: Dict[str, List[int]] = {}
        for sn in [Macros.val, Macros.test]:
            sn2ids[sn] = su.io.load(split_dir / f"{sn}.json")
        all_ids: Set[int] = set(sum(sn2ids.values(), []))

        # Load data
        with tqdm("Loading data") as pbar:
            dataset: List[TData] = load_dataset(
                data_dir, clz=self.data_cls, expected_ids=all_ids, pbar=pbar
            )
        indexed_dataset: Dict[int, TData] = {d.id: d for d in dataset}

        sn2ds = {sn: [indexed_dataset[i] for i in ids] for sn, ids in sn2ids.items()}

        for sn in [Macros.val]:
            su.io.mkdir(out_dir / sn)
            save_dataset(out_dir / sn, sn2ds[sn])

    def prepare_test(
        self,
        data_dir: Path_drw,
        split_dir: Path_drw,
        out_dir: Path_dc,
    ):
        """
        Prepares the evaluation test sets under the requested setup because test data are special (mimic the real-world usage).

        :param data_dir: source data directory.
        :param split_dir: split ids directory.
        :param out_dir: output data directory, which will contain val/ and test/ datasets.
        :param seed: random seed (for sampling data).
        """
        eval_config = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "data_dir", "split_dir", "out_dir"]
        }
        self.config.update(eval_config)
        logger.info(f"Config:\n{self.config}")

        data_dir = Path(data_dir.abs_path)
        split_dir = Path(split_dir.abs_path)
        out_dir = Path(out_dir.abs_path)

        su.io.mkdir(out_dir, fresh=True)
        su.io.dump(out_dir / "config.json", self.config, su.io.Fmt.jsonNoSort)

        # Load data ids
        sn2ids: Dict[str, List[int]] = {}
        for sn in [Macros.test]:
            sn2ids[sn] = su.io.load(split_dir / f"{sn}.json")
        all_ids: Set[int] = set(sum(sn2ids.values(), []))

        # Load data
        with tqdm("Loading data") as pbar:
            dataset: List[TData] = load_dataset(
                data_dir, clz=self.data_cls, expected_ids=all_ids, pbar=pbar
            )
        indexed_dataset: Dict[int, TData] = {d.id: d for d in dataset}

        sn2ds = {sn: [indexed_dataset[i] for i in ids] for sn, ids in sn2ids.items()}

        for sn in [Macros.test]:
            su.io.mkdir(out_dir / sn)
            save_dataset(out_dir / sn, sn2ds[sn])

    def prepare_subset_test(
        self,
        data_dir: Path_drw,
        split_dir: Path_drw,
        out_dir: Path_dc,
    ):
        """
        Prepares the evaluation test sets under the requested setup, this is the subset because in exlong we only 
        eval on the subset of the dataset.

        :param data_dir: source data directory.
        :param split_dir: split ids directory.
        :param out_dir: output data directory, which will contain val/ and test/ datasets.
        :param seed: random seed (for sampling data).
        """
        eval_config = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "data_dir", "split_dir", "out_dir"]
        }
        self.config.update(eval_config)
        logger.info(f"Config:\n{self.config}")

        data_dir = Path(data_dir.abs_path)
        split_dir = Path(split_dir.abs_path)
        out_dir = Path(out_dir.abs_path)

        su.io.mkdir(out_dir, fresh=True)
        su.io.dump(out_dir / "config.json", self.config, su.io.Fmt.jsonNoSort)

        # Load data ids
        sn2ids: Dict[str, List[int]] = {}
        for sn in [Macros.test]:
            sn2ids[sn] = su.io.load(split_dir / f"{sn}.json")

        # Load data
        with tqdm("Loading data") as pbar:
            dataset: List[TData] = load_dataset(
                data_dir, clz=self.data_cls, pbar=pbar
            )

        for sn in [Macros.test]:
            su.io.mkdir(out_dir / sn)
            save_dataset(out_dir / sn, dataset)

    def prepare_statement_eval(
        self,
        data_dir: Path_drw,
        split_dir: Path_drw,
        out_dir: Path_dc,
        min_stmt: int = 0,
        max_eval_data: int = 100_000,
        max_per_proj: int = -1,
        seed: int = time.time_ns(),
    ):
        """Prepare the evaluation data for statement-level evaluation."""

        assert min_stmt >= 0
        assert max_eval_data > 0
        assert max_per_proj > 0 or max_per_proj == -1

        eval_config = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "data_dir", "split_dir", "out_dir"]
        }
        self.config.update(eval_config)
        logger.info(f"Config:\n{self.config}")

        data_dir = Path(data_dir.abs_path)
        split_dir = Path(split_dir.abs_path)
        out_dir = Path(out_dir.abs_path)

        su.io.mkdir(out_dir, fresh=True)
        su.io.dump(out_dir / "config.json", self.config, su.io.Fmt.jsonNoSort)

        # Load data ids
        sn2ids: Dict[str, List[int]] = {}
        for sn in [Macros.val, Macros.test]:
            try:
                sn2ids[sn] = su.io.load(split_dir / f"{sn}.json")
            except FileNotFoundError:
                sn2ids[sn] = []
        all_ids: Set[int] = set(sum(sn2ids.values(), []))

        # Load data
        with tqdm("Loading data") as pbar:
            dataset: List[TData] = load_dataset(
                data_dir, clz=self.data_cls, expected_ids=all_ids, pbar=pbar
            )

        indexed_dataset: Dict[str, TData] = {d.id: d for d in dataset}

        sn2ds = {sn: [indexed_dataset[i] for i in ids] for sn, ids in sn2ids.items()}

        for sn in [Macros.val, Macros.test]:
            # Find all available eval data_id: (data_id, stmt_i) tuples
            id2locs: Dict[int, List[Tuple[str, int]]] = collections.defaultdict(list)
            eval_locs = []

            for data in sn2ds[sn]:
                for stmt_i, stmt in enumerate(data.test_stmt_toks):
                    if stmt_i < self.config.min_stmt:
                        continue
                    eval_locs.append((data.id, stmt_i))
                    id2locs[data.id].append((data.id, stmt_i))

            # Assemble the final eval set of data and gold
            eval_ds: List[TData] = []
            gold_stmts: List[List[str]] = []
            gold_insns: List[List[str]] = []
            gold_fqinsns: List[List[str]] = []
            eval_locs.sort()
            for data_id, stmt_i in eval_locs:
                data = copy.deepcopy(indexed_dataset[data_id])
                gold_stmt = data.test_stmt_toks[stmt_i]
                data.cutoff(stmt_i)
                eval_ds.append(data)
                gold_stmts.append(gold_stmt)

            # Save
            su.io.mkdir(out_dir / sn)
            save_dataset(out_dir / sn, eval_ds)
            su.io.dump(out_dir / sn / "gold_stmts.jsonl", gold_stmts)
            su.io.dump(out_dir / sn / "gold_insns.jsonl", gold_insns)
            su.io.dump(out_dir / sn / "gold_fqinsns.jsonl", gold_fqinsns)
            su.io.dump(out_dir / sn / "eval_locs.jsonl", eval_locs)


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(EvalSetup, as_positional=False)
