import collections
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import seutil as su
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw, Path_fr
from seutil.project import Project
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from etestgen.data.data import DataMUT2E
from etestgen.data.utils import load_dataset
from etestgen.macros import Macros

logger = su.log.get_logger(__name__)


class GetSplit:
    def random_split_from_repos(
        self,
        data_dir: Path,
        out_dir: Path,
        seed: int = 7,
    ):
        random.seed(seed)
        # load data
        with tqdm(desc="loading data") as pbar:
            dataset: List[DataMUT2E] = load_dataset(
                data_dir, clz=DataMUT2E, only=["project"], pbar=pbar
            )

        project_list = []
        for d in dataset:
            project_list.append(d.project)

        # split projects to train/val/test
        project_list = list(set(project_list))
        project_list.sort()
        random.shuffle(project_list)
        train_projects, val_test_projects = train_test_split(
            project_list, test_size=0.1, random_state=seed
        )
        val_projects, test_projects = train_test_split(
            val_test_projects, test_size=0.51, random_state=seed
        )
        proj2sn: Dict[str, str] = {p: Macros.train for p in train_projects}
        for p in test_projects:
            proj2sn[p] = Macros.test
        for p in val_projects:
            proj2sn[p] = Macros.val

        sn2ids: Dict[str, List[str]] = collections.defaultdict(list)
        for d in dataset:
            sn2ids[proj2sn[d.project]].append(d.id)

        # save split
        su.io.dump(out_dir / "train-projects.json", train_projects)
        su.io.dump(out_dir / "valid-projects.json", val_projects)
        su.io.dump(out_dir / "test-projects.json", test_projects)
        su.io.mkdir(out_dir)
        for sn, ids in sn2ids.items():
            print(f"{sn}: {len(ids)}")
            su.io.dump(out_dir / f"{sn}.json", ids)

    def split_data_by_repos(
        self,
        data_dir: Path,
        out_dir: Path,
        seed: int = 7,
    ):
        """Given the split of repos, split the data accordingly."""
        random.seed(seed)
        # load data
        with tqdm(desc="loading data") as pbar:
            dataset: List[DataMUT2E] = load_dataset(
                data_dir, clz=DataMUT2E, only=["project"], pbar=pbar
            )

        train_projects = su.io.load(
            Macros.results_dir / "repos" / "train-projects.json"
        )
        val_projects = su.io.load(Macros.results_dir / "repos" / "valid-projects.json")
        test_projects = su.io.load(Macros.results_dir / "repos" / "test-projects.json")

        proj2sn: Dict[str, str] = {p: Macros.train for p in train_projects}
        for p in test_projects:
            proj2sn[p] = Macros.test
        for p in val_projects:
            proj2sn[p] = Macros.val

        sn2ids: Dict[str, List[str]] = collections.defaultdict(list)
        for d in dataset:
            sn2ids[proj2sn[d.project]].append(d.id)

        # save split
        su.io.dump(out_dir / "train-projects.json", train_projects)
        su.io.dump(out_dir / "valid-projects.json", val_projects)
        su.io.dump(out_dir / "test-projects.json", test_projects)
        su.io.mkdir(out_dir)
        for sn, ids in sn2ids.items():
            print(f"{sn}: {len(ids)}")
            su.io.dump(out_dir / f"{sn}.json", ids)

    def get_split_from_repos(
        self,
        repos_file: Path_fr,
        data_dir: Path_drw,
        out_dir: Union[Path_drw, Path_dc],
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        limit_test: Optional[int] = None,
        seed: int = 7,
    ):
        repos_file = Path(repos_file.abs_path)
        data_dir = Path(data_dir.abs_path)
        out_dir = Path(out_dir.abs_path)
        random.seed(seed)

        # load projects
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        proj2sn: Dict[str, str] = {p.full_name: p.data["sources"] for p in projects}

        # load data
        with tqdm(desc="loading data") as pbar:
            dataset: List[DataMUT2E] = load_dataset(
                data_dir, clz=DataMUT2E, only=["project"], pbar=pbar
            )
        sn2ids: Dict[str, List[str]] = collections.defaultdict(list)
        for d in dataset:
            if d.project not in proj2sn:
                # project removed after process_raw_data
                continue
            sn2ids[proj2sn[d.project]].append(d.id)

        # potentially limit data
        if limit_train is not None:
            random.shuffle(sn2ids[Macros.train])
            sn2ids[Macros.train] = sn2ids[Macros.train][:limit_train]
            sn2ids[Macros.train].sort()
        if limit_val is not None:
            random.shuffle(sn2ids[Macros.val])
            sn2ids[Macros.val] = sn2ids[Macros.val][:limit_val]
            sn2ids[Macros.val].sort()
        if limit_test is not None:
            random.shuffle(sn2ids[Macros.test])
            sn2ids[Macros.test] = sn2ids[Macros.test][:limit_test]
            sn2ids[Macros.test].sort()

        # save split
        su.io.mkdir(out_dir)
        for sn, ids in sn2ids.items():
            print(f"{sn}: {len(ids)}")
            su.io.dump(out_dir / f"{sn}.json", ids)


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(GetSplit, as_positional=False)
