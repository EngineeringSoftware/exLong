import random
import sys
from pathlib import Path

import seutil as su
from invoke import task

from etestgen.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)


setup_2_data_cls = {
    "mut2e-with-name-ft": "MUT2E",
    "ne2e-with-name-ft": "NE2E",
    "conditionne2e-with-name-ft": "NE2E",
    "conditionnestack2e-no-name-ft": "NE2E",
    "conditionnestack2e-all-no-name-ft": "NE2E",
    "conditionnestack2e-with-name-ft": "NE2E",
    "conditionnestack2e-all-with-name-ft": "NE2E",
    "conditionnestack2e-sample-with-name-ft": "NE2E",
}


@task
def setup_model_data(
    c, setup_name: str, source_setup_name: str = "ne2e", debug: bool = False
):
    """
    Split the dataset into train/val/test sets, the test data will be seperate into eval/ and real-eval/.
    """

    if debug:
        setup_name += "-debug"
        split_arg = "--limit_train 800 --limit_val 100 --limit_test 100 --seed 7"
    else:
        split_arg = "--seed 7"

    setup_dir = Macros.work_dir / "setup" / setup_name

    data_dir = Macros.work_dir / "data" / source_setup_name
    split_dir = setup_dir / "split"
    data_cls = setup_2_data_cls[setup_name]

    su.io.mkdir(setup_dir, fresh=True)

    # get split
    c.run(
        f"""python -m etestgen.data.get_split\
            split_data_by_repos\
            --data_dir {data_dir}\
            --out_dir {split_dir}\
            {split_arg} """
    )

    # prepare train
    c.run(
        f"""python -m etestgen.data.eval_setup\
            --data_cls {data_cls}\
            prepare_train\
            --data_dir {data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/train """
    )

    # prepare eval
    c.run(
        f"""python -m etestgen.data.eval_setup\
            --data_cls {data_cls}\
            prepare_eval\
            --data_dir {data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/eval """
    )
    c.run(
        f"""python -m etestgen.data.eval_setup\
            --data_cls {data_cls}\
            prepare_test\
            --data_dir {data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/eval """
    )
    test_data_dir = Macros.work_dir / "data" / "rq1-eval"
    # prepare test
    c.run(
        f"""python -m etestgen.data.eval_setup\
            --data_cls {data_cls}\
            prepare_subset_test\
            --data_dir {test_data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/real-eval """
    )


@task
def process_codellama_data(c, setup_name: str):
    c.run(
        f"""python -m etestgen.codellama.DataProcessor\
            --config_file configs/codellama-7b-{setup_name}.yaml\
            process_train_data"""
    )
    c.run(
        f"""python -m etestgen.codellama.DataProcessor\
            --config_file configs/codellama-7b-{setup_name}.yaml\
            process_test_data"""
    )
    if "stack" in setup_name:
        c.run(
            f"""python -m etestgen.codellama.DataProcessor\
                --config_file configs/codellama-7b-{setup_name}.yaml\
                process_real_test_data"""
        )
