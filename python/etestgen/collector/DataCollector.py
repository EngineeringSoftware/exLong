from cgi import test
from typing import *
import seutil as su
from seutil.project import Project
from pathlib import Path
from seutil.maven import MavenModule, MavenProject
from tqdm import tqdm
from jsonargparse import CLI
from collections import defaultdict
import numpy as np
import random

random.seed(42)

from etestgen.data.tool import Tool
from etestgen.macros import Macros
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.data.data import TData, parse_data_cls

from etestgen.data.data import DataNE2E
from etestgen.eval.compute_throws_coverage import (
    Coverage,
    TestMethod,
)

logger = su.log.get_logger(__name__)


class DataCollector:
    def __init__(self, repo_file_name = "repos.json") -> None:
        self.repo_file = Macros.work_dir / "repos" / "filtered" / repo_file_name
        self.coverage_dir = Macros.work_dir / "coverage-new"

    def collect_non_cover_ne2e_data(self, source_dataset: str):
        """
        Collect dataset for non covered methods. (rq2)
        + Mut
        + test file
        + netest
        + stack trace (w.o. line number)
        + condition
        """
        test_class_2_context = {}
        test_class_2_methods = defaultdict(list)
        project_2_test_classes = defaultdict(set)
        current_test_class_name = ""

        # helper functions
        def find_existing_test_classes(project_name: str, module_i: int):
            """
            Collect the existing test classes in the target module.
            Collect the existing test methods in the target module.
            """
            nonlocal test_class_2_context
            nonlocal test_class_2_methods
            nonlocal project_2_test_classes

            tests_file = su.io.load(
                Macros.work_dir
                / "coverage-new"
                / project_name
                / str(module_i)
                / "manual.tests.jsonl",
                clz=TestMethod,
            )
            test_class_names = set()
            for test in tests_file:
                test_class_names.add(test.cname)
                test_class_2_context[f"{project_name}.{module_i}.{test.cname}"] = (
                    test.ccontext
                )
                test_class_2_methods[f"{project_name}.{module_i}.{test.cname}"].append(
                    test
                )
            # update dict
            project_2_test_classes[f"{project_name}.{module_i}"].update(
                test_class_names
            )

        def extract_test_file_context(
            project_name: str, module_i: int, mut_key: str
        ) -> Tuple[str, TestMethod]:
            """
            Extract the test file context, and netest within given the Test Name by name matching.
            """

            nonlocal current_test_class_name
            mut_class_name = mut_key.split("#")[0]
            mut_class_no_innerclass = mut_key.split("#")[0].split("$")[0]
            if (
                f"{mut_class_name}Test"
                in project_2_test_classes[f"{project_name}.{module_i}"]
            ):
                test_file_context = test_class_2_context[
                    f"{project_name}.{module_i}.{mut_class_name}Test"
                ]
                current_test_class_name = f"{mut_class_name}Test"
                gold_test_method = test_class_2_methods[
                    f"{project_name}.{module_i}.{mut_class_name}Test"
                ][0]
            elif (
                f"{mut_class_no_innerclass}Test"
                in project_2_test_classes[f"{project_name}.{module_i}"]
            ):
                test_file_context = test_class_2_context[
                    f"{project_name}.{module_i}.{mut_class_no_innerclass}Test"
                ]
                gold_test_method = test_class_2_methods[
                    f"{project_name}.{module_i}.{mut_class_no_innerclass}Test"
                ][0]
                current_test_class_name = f"{mut_class_no_innerclass}Test"
            else:
                test_file_context = None
                current_test_class_name = ""
                gold_test_method = None
            return test_file_context, gold_test_method

        def extract_netest(
            mut_key: str, project_name: str, module_i: int
        ) -> List[TestMethod]:
            ne_tests = []
            mut_key = mut_key.replace(".", "/")
            if not mut2test or len(mut2test[mut_key]) == 0:
                if current_test_class_name != "":
                    # find test methods within the test class
                    existing_methods = test_class_2_methods[
                        f"{project_name}.{module_i}.{current_test_class_name}"
                    ]
                    ne_tests = [m for m in existing_methods if m.pattern is None]
            else:
                ne_tests = [
                    test_methods[tid]
                    for tid in mut2test[mut_key]
                    if test_methods[tid].pattern is None
                ]
            #
            return ne_tests

        # load collected dataset
        dataset = load_dataset(Macros.data_dir / source_dataset, clz=DataNE2E)
        module_2_test_coverage = {}
        new_dataset = []
        # stats
        no_netest, no_test_context = 0, 0
        with tqdm(total=len(dataset), desc="Collecting non covered data") as pbar:
            for dt in dataset:
                project_name, module_i = dt.project, dt.module_i
                if f"{project_name}.{module_i}" not in project_2_test_classes:
                    find_existing_test_classes(project_name, module_i)
                #
                test_file_context, gold_test = extract_test_file_context(
                    project_name, module_i, dt.mut_key
                )
                # netest
                if f"{project_name}.{module_i}" not in module_2_test_coverage:
                    module_2_test_coverage[f"{project_name}.{module_i}"] = (
                        extract_module_test_coverage(
                            dt, coverage_out_dir=Macros.work_dir / "coverage-new"
                        )
                    )
                mut2test, test_methods = module_2_test_coverage[
                    f"{project_name}.{module_i}"
                ]
                ne_tests = extract_netest(dt.mut_key, project_name, module_i)
                dt.test_ne = [t.raw_code for t in ne_tests]
                if not test_file_context and len(ne_tests) > 0:
                    dt.test_context = [t.ccontext for t in ne_tests]
                    dt.test_method = ne_tests[0]
                else:
                    dt.test_context = test_file_context
                    dt.test_method = gold_test
                # stats
                if len(ne_tests) == 0:
                    no_netest += 1

                if dt.test_context is None or len(dt.test_context) == 0:
                    no_test_context += 1
                else:
                    assert dt.test_method is not None, "No at least one ne-test"
                    new_dataset.append(dt)
                pbar.update(1)

        logger.info(f"No netest: {no_netest}\n No test context: {no_test_context}")
        save_dataset(Macros.data_dir / f"new-{source_dataset}", new_dataset)

    def collect_netest_data(self, target_projects: List[str] = None):
        """
        For each etest, collect the netest that cover the same method.
        """
        # load mut2e data
        mut2e_data_dir = Macros.work_dir / "data" / "mut2e"
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(
                mut2e_data_dir, clz=parse_data_cls("MUT2E"), pbar=pbar
            )

        ne2e_data_list = []
        normal_test_cover_mut, no_normal_test = 0, 0
        module_2_tests = {}
        ne2e_data_dir = Macros.data_dir / "ne2e-new"
        #
        with tqdm(total=len(mut2e_dataset), desc="Pairing etest with netest") as pbar:
            for i, mut2e_dt in enumerate(mut2e_dataset):
                cur_prj = mut2e_dt.project
                if target_projects and cur_prj not in target_projects:
                    continue
                mut = mut2e_dt.mut_key
                cur_module = mut2e_dt.module_i
                if f"{cur_prj}.{cur_module}" not in module_2_tests:
                    module_2_tests[f"{cur_prj}.{cur_module}"] = (
                        extract_module_test_coverage(
                            mut2e_dt, coverage_out_dir=Macros.work_dir / "coverage-new"
                        )
                    )
                #
                mut2test, test_methods = module_2_tests[f"{cur_prj}.{cur_module}"]
                ne_tests = []
                mut_key = mut.replace(".", "/")
                if not mut2test:
                    no_normal_test += 1
                else:
                    if len(mut2test[mut_key]) == 0:
                        no_normal_test += 1
                    else:
                        ne_tests = [
                            tid
                            for tid in mut2test[mut_key]
                            if test_methods[tid].pattern is None
                        ]
                        if len(ne_tests) > 0:
                            normal_test_cover_mut += 1
                        else:
                            no_normal_test += 1
                data = DataNE2E(**mut2e_dt.__dict__)
                for ne_id in ne_tests:
                    test_ne = test_methods[ne_id]
                    data.netest_sign.append(test_ne.ast.get_sign())
                    data.test_ne.append(test_ne.raw_code)
                    data.netest_context.append(test_ne.context)
                    data.test_ne_key.append(
                        test_ne.cname.replace(".", "/") + "#" + test_ne.mname + "#()V"
                    )
                #
                data.id = f"ne2e-{i}"
                ne2e_data_list.append(data)
                pbar.update(1)
        logger.info(
            f"Normal test cover mut: {normal_test_cover_mut}\n No normal tests cover the same method: {no_normal_test}"
        )
        save_dataset(ne2e_data_dir, ne2e_data_list, append=True)

    def sample_ne2e_data(
        self,
        sample_size: int = 1,
    ):
        """
        Sample ne test from ne2e full dataset. NOTE: pick the shortest top k ne test
        """

        data_dir = Macros.work_dir / "data" / f"ne2e-{sample_size}"
        su.io.mkdir(data_dir, fresh=True)
        with tqdm("Loading NE2E data") as pbar:
            ne2e_dataset = load_dataset(
                Macros.work_dir / "data" / "ne2e",
                clz=parse_data_cls("NE2E"),
                pbar=pbar,
            )

        sample_ne2e_dataset = []
        with tqdm(total=len(ne2e_dataset), desc="Sampling NE2E data") as pbar:
            for ne2e_dt in ne2e_dataset:
                netest_length = np.array([len(t.split()) for t in ne2e_dt.test_ne])
                sorted_ids = np.argsort(netest_length)[:sample_size]  # the shortest ids
                # create new ne2e data
                ne2e_dt.netest_sign = [
                    ne2e_dt.netest_sign[shortest_id] for shortest_id in sorted_ids
                ]
                ne2e_dt.test_ne = [
                    ne2e_dt.test_ne[shortest_id] for shortest_id in sorted_ids
                ]
                ne2e_dt.netest_context = [
                    ne2e_dt.netest_context[shortest_id] for shortest_id in sorted_ids
                ]
                ne2e_dt.test_ne_key = [
                    ne2e_dt.test_ne_key[shortest_id] for shortest_id in sorted_ids
                ]
                sample_ne2e_dataset.append(ne2e_dt)
                pbar.update(1)
        #
        save_dataset(data_dir, sample_ne2e_dataset, append=True)

    def extract_ne2e_data(
        self,
        data_dir: su.arg.RPath = Macros.work_dir / "data" / "ne2e",
        sample_one: bool = False,
    ):
        """
        Extract NE2E dataset.
        If sample_one is True, then only sample 1 netest for each etest.
        """

        # load mut2e data
        mut2e_data_dir = Macros.work_dir / "data" / "mut2e"
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(
                mut2e_data_dir, clz=parse_data_cls("MUT2E"), pbar=pbar
            )
        cur_prj, cur_module, test2mut, test_methods = None, None, None, None
        data_list = []
        logs = {
            "no_mut_dict": set(),
            "random_sample_ne": set(),
            "no_normal_tests": set(),
            "normal_test_cover_mut": 0,
            "normal_test_in_same_module": 0,
            "normal_test_in_same_class": 0,
        }
        normal_test_cover_mut, normal_test_same_class, normal_test_same_module = 0, 0, 0

        module_2_tests = {}
        with tqdm(total=len(mut2e_dataset), desc="Pairing etest with netest") as pbar:
            for i, mut2e_dt in enumerate(mut2e_dataset):
                mut = mut2e_dt.mut_key
                cur_prj = mut2e_dt.project
                cur_module = mut2e_dt.module_i
                if f"{cur_prj}.{cur_module}" not in module_2_tests:
                    module_2_tests[f"{cur_prj}.{cur_module}"] = prepare_module_data(
                        mut2e_dt
                    )
                #

                test2mut, test_methods = module_2_tests[f"{cur_prj}.{cur_module}"]
                mut2test = defaultdict(list)

                ne_tests = []
                if not test2mut:
                    # fail to extract the test 2 mut mapping, sample 5 tests
                    logs["no_mut_dict"].add((cur_prj, cur_module))
                    ne_tests = [
                        i for i, tm in enumerate(test_methods) if tm.pattern is None
                    ]
                    ne_tests = random.sample(ne_tests, min(5, len(ne_tests)))
                    if len(ne_tests) > 0:
                        normal_test_same_module += 1
                else:
                    for k, v in test2mut.items():
                        mut2test[v].append(int(k))

                    if len(mut2test[mut]) == 0:
                        # can not find the ntest for the mut
                        ne_tests = find_netests_in_same_class(
                            test_methods, mut2e_dt.test_e_key
                        )
                        if len(ne_tests) > 0:
                            normal_test_same_class += 1
                    else:
                        ne_tests = [
                            tid
                            for tid in mut2test[mut]
                            if test_methods[tid].pattern is None
                        ]
                        if len(ne_tests) > 0:
                            normal_test_cover_mut += 1

                    if len(ne_tests) == 0:
                        # sample test from the same modules
                        ne_tests = [
                            i for i, tm in enumerate(test_methods) if tm.pattern is None
                        ]
                        ne_tests = random.sample(ne_tests, min(5, len(ne_tests)))
                        if len(ne_tests) > 0:
                            logs["random_sample_ne"].add((cur_prj, cur_module))
                            normal_test_same_module += 1

                if len(ne_tests) == 0:
                    logs["no_normal_tests"].add((cur_prj, cur_module))

                data = DataNE2E(**mut2e_dt.__dict__)
                if sample_one and ne_tests:
                    ne_tests = [random.choice(ne_tests)]
                for ne_id in ne_tests:
                    test_ne = test_methods[ne_id]
                    data.netest_sign.append(test_ne.ast.get_sign())
                    data.test_ne.append(test_ne.raw_code)
                    data.netest_context.append(test_ne.context)
                    data.test_ne_key.append(
                        test_ne.cname.replace(".", "/") + "#" + test_ne.mname + "#()V"
                    )

                data.id = f"ne2e-{i}"
                data_list.append(data)
                if i % 1000 == 0:
                    save_dataset(data_dir, data_list, append=True)
                    data_list = []
                pbar.update(1)
        save_dataset(data_dir, data_list, append=True)
        logs["normal_test_cover_mut"] = normal_test_cover_mut
        logs["normal_test_in_same_module"] = normal_test_same_module
        logs["normal_test_in_same_class"] = normal_test_same_class
        su.io.dump(data_dir / "collection-logs.json", logs)

    def add_pattern_mut2e_dataset(self):
        """
        Add etest pattern to mut2e dataset

        1. Load dataset
        2. find the test file according to project.module.key
        3. replace with new etest
        """

        mut2e_data_dir = Macros.work_dir / "data" / "mut2e"
        mut2d_new_data_dir = Macros.work_dir / "data" / "mut2e-new"
        su.io.mkdir(mut2d_new_data_dir, fresh=True)
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(
                mut2e_data_dir, clz=parse_data_cls("MUT2E"), pbar=pbar
            )
        new_mut2e_dataset = []
        module2tests = {}
        for mut2e_dt in tqdm(mut2e_dataset, total=len(mut2e_dataset)):
            project_name = mut2e_dt.project
            module_i = mut2e_dt.module_i
            key2pattern = {}
            if f"{project_name}.{module_i}" not in module2tests:
                module2tests[f"{project_name}.{module_i}"] = key2pattern
                # prepare mapping
                coverage_dir = (
                    Macros.work_dir / "coverage" / project_name / str(module_i)
                )
                manual_test_list = su.io.load(
                    coverage_dir / "manual.tests.jsonl", clz=TestMethod
                )
                for test in manual_test_list:
                    key = (
                        test.cname.replace(".", "/")
                        + "#"
                        + test.mname
                        + "#()V"
                        + str(test.exception)
                    )
                    try:
                        assert key not in key2pattern
                    except AssertionError:
                        print("Duplicate test key!")
                        save_dataset(mut2d_new_data_dir, new_mut2e_dataset, append=True)
                        breakpoint()
                    key2pattern[key] = test.pattern
                #
            else:
                key2pattern = module2tests[f"{project_name}.{module_i}"]

            key = mut2e_dt.test_e_key + mut2e_dt.etype
            if key not in key2pattern:
                save_dataset(mut2d_new_data_dir, new_mut2e_dataset, append=True)
                print("Can not find the target test!")
                breakpoint()
            else:
                mut2e_dt.etest_pattern = key2pattern[key]
                new_mut2e_dataset.append(mut2e_dt)
        save_dataset(mut2d_new_data_dir, new_mut2e_dataset, append=True)

    def fix_mut2e_dataset(self):
        """
        Fix try catch pattern in existing mut2e dataset.

        1. Load dataset
        2. find the test file according to project.module.key
        3. replace with new etest
        """

        mut2e_data_dir = Macros.work_dir / "data" / "mut2e-trycatchbug"
        mut2d_new_data_dir = Macros.work_dir / "data" / "mut2e-new"
        su.io.mkdir(mut2d_new_data_dir)
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(
                mut2e_data_dir, clz=parse_data_cls("MUT2E"), pbar=pbar
            )
        new_mut2e_dataset = []
        module2tests = {}
        for mut2e_dt in tqdm(mut2e_dataset, total=len(mut2e_dataset)):
            project_name = mut2e_dt.project
            module_i = mut2e_dt.module_i
            key2test = {}
            if f"{project_name}.{module_i}" not in module2tests:
                module2tests[f"{project_name}.{module_i}"] = key2test
                # prepare mapping
                coverage_dir = (
                    Macros.work_dir / "coverage" / project_name / str(module_i)
                )
                manual_test_list = su.io.load(
                    coverage_dir / "manual.tests.jsonl", clz=TestMethod
                )
                for test in manual_test_list:
                    key = (
                        test.cname.replace(".", "/")
                        + "#"
                        + test.mname
                        + "#()V"
                        + str(test.exception)
                    )
                    try:
                        assert key not in key2test
                    except AssertionError:
                        print("Duplicate test key!")
                        save_dataset(mut2d_new_data_dir, new_mut2e_dataset, append=True)
                        breakpoint()
                    key2test[key] = test.raw_code
                #
            else:
                key2test = module2tests[f"{project_name}.{module_i}"]

            key = mut2e_dt.test_e_key + mut2e_dt.etype
            if key not in key2test:
                save_dataset(mut2d_new_data_dir, new_mut2e_dataset, append=True)
                print("Can not find the target test!")
                breakpoint()
            else:
                test_raw_code = key2test[key]
                mut2e_dt.test_e = test_raw_code
                new_mut2e_dataset.append(mut2e_dt)
        save_dataset(mut2d_new_data_dir, new_mut2e_dataset, append=True)

    def fix_call_stack(self):
        mut2e_data_dir = Macros.work_dir / "data" / "mut2e"
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(
                mut2e_data_dir, clz=parse_data_cls("MUT2E"), pbar=pbar
            )

        for dt in tqdm(mut2e_dataset, total=len(mut2e_dataset)):
            method_calls = "```java\n"
            called_ms_seq = []
            for ms_call in dt.call_stacks:
                if ms_call.code is None:
                    continue
                if ms_call.code not in called_ms_seq:
                    called_ms_seq.append(ms_call)
            #
            #
            if len(called_ms_seq) == 0:
                print(f"No call stack! {dt.id}")
                breakpoint()
            if len(called_ms_seq) == 1:
                called_ms = called_ms_seq[0]
                if called_ms.name + "#" + called_ms.desc == "#".join(
                    dt.mut_key.split("#")[1:]
                ):
                    pass
                else:
                    print(f"Only one call stack! not mut {dt.id}")

    def seperate_nestack2e_dataset(self):
        """
        Seperate nestack2e dataset into two parts: with stack trace and without stack traces.
        """

        nestack2e_test_data_dir = Macros.work_dir / "data" / "nestack2e-test"
        nestack2e_all = load_dataset(
            nestack2e_test_data_dir, clz=parse_data_cls("NESTACK2E")
        )
        nestack2e_no_stack_trace_dir = (
            Macros.work_dir / "data" / "nestack2e-no-stack-trace"
        )
        nestack_w_stack_trace_dir = (
            Macros.work_dir / "data" / "nestack2e-with-stack-trace"
        )
        data_with_stack_trace = []
        data_without_stack_trace = []

        for dt in nestack2e_all:
            if (
                len(dt.stack_traces) == 0
                and len(dt.call_stacks) == 1
                and dt.call_stacks[0].namedesc == "#".join(dt.mut_key.split("#")[1:])
            ):
                data_with_stack_trace.append(dt)
            elif len(dt.stack_traces) > 0:
                data_with_stack_trace.append(dt)
            else:
                data_without_stack_trace.append(dt)
        #
        save_dataset(nestack_w_stack_trace_dir, data_with_stack_trace)
        save_dataset(nestack2e_no_stack_trace_dir, data_without_stack_trace)

    def construct_nestack2e_testset(self):
        """
        For each cov2e data, add the stack traces collected from netests.
        """
        cov2e_data_dir = Macros.work_dir / "data" / "cov2e"
        nestack2e_data_dir = Macros.work_dir / "data" / "nestack2e"
        target_data_dir = Macros.work_dir / "data" / "nestack2e-test"
        with tqdm("Loading Cov2e data") as pbar:
            cov2e_dataset = load_dataset(
                cov2e_data_dir, clz=parse_data_cls("NE2E"), pbar=pbar
            )
        with tqdm("Loading nestack2e dataset") as pbar:
            nestack2e_dataset = load_dataset(
                nestack2e_data_dir, clz=parse_data_cls("NESTACK2E"), pbar=pbar
            )

        test_projects = [dt.project for dt in nestack2e_dataset]
        new_testdata_with_stack_trace = []
        no_stack_trace = 0
        #
        with tqdm(
            total=len(cov2e_dataset),
            desc="Fix the call stacks with the call stack from netest",
        ) as pbar:
            for i, cov2e_dt in enumerate(cov2e_dataset):
                if cov2e_dt.project not in test_projects:
                    continue
                cov2e_dt.stack_traces = set()
                cur_prj = cov2e_dt.project
                for nestack_dt in nestack2e_dataset:
                    if (
                        nestack_dt.project == cur_prj
                        and nestack_dt.etype == cov2e_dt.etype
                    ):
                        for i, ms in enumerate(nestack_dt.call_stacks):
                            mskey = ms.key.replace(".", "/")
                            if cov2e_dt.mut_key == mskey:
                                cov2e_dt.stack_traces.add(
                                    tuple(nestack_dt.call_stacks[i:])
                                )
                cov2e_dt.stack_traces = list(cov2e_dt.stack_traces)
                new_testdata_with_stack_trace.append(cov2e_dt)
                if len(cov2e_dt.stack_traces) == 0:
                    no_stack_trace += 1
                pbar.update(1)
        logger.info(
            f"In total {len(new_testdata_with_stack_trace)} data for eval, among them {no_stack_trace} data has no stack traces from netests."
        )
        save_dataset(target_data_dir, new_testdata_with_stack_trace)

    def sanity_check_rq1_eval(self):
        """
        Sanity check the evaluation dataset: e_stack_trace only has one stack trace.
        """

        rq1_eval = load_dataset(Macros.data_dir / "rq1-eval", clz=DataNE2E)
        new_rq1_eval = []
        for dt in rq1_eval:
            m_name = dt.e_stack_trace[-1][0]["method"]
            assert len(dt.e_stack_trace) > 0, "No stack trace is extracted"
            if dt.id in [
                "ne2e-106",
                "ne2e-107",
                "ne2e-108",
                "ne2e-115",
                "ne2e-122",
                "ne2e-109",
                "ne2e-119",
                "ne2e-110",
                "ne2e-111",
                "ne2e-112",
                "ne2e-113",
                "ne2e-121",
                "ne2e-114",
                "ne2e-116",
                "ne2e-117",
                "ne2e-118",
                "ne2e-120",
                "ne2e-123",
                "ne2e-124",
                "ne2e-1022",
                "ne2e-1023",
                "ne2e-1024",
                "ne2e-1026",
                "ne2e-1027",
                "ne2e-1028",
                "ne2e-1029",
                "ne2e-1030",
                "ne2e-1031",
                "ne2e-1032",
                "ne2e-1033",
                "ne2e-1034",
                "ne2e-1036",
                "ne2e-1039",
                "ne2e-1040",
                "ne2e-1835",
                "ne2e-1836",
                "ne2e-1930",
                "ne2e-1932",
                "ne2e-1933",
                "ne2e-1934",
                "ne2e-2134",
                "ne2e-2135",
                "ne2e-2137",
                "ne2e-2138",
                "ne2e-2910",
                "ne2e-2911",
                "ne2e-2912",
                "ne2e-2913",
                "ne2e-2914",
                "ne2e-2917",
                "ne2e-2922",
                "ne2e-2924",
                "ne2e-2925",
                "ne2e-2926",
                "ne2e-3038",
                "ne2e-3039",
                "ne2e-3091",
                "ne2e-3092",
                "ne2e-3093",
                "ne2e-5308",
                "ne2e-5309",
                "ne2e-5310",
                "ne2e-5311",
                "ne2e-5312",
                "ne2e-5313",
                "ne2e-5353",
                "ne2e-6480",
                "ne2e-6481",
                "ne2e-6483",
                "ne2e-6485",
                "ne2e-6487",
                "ne2e-6482",
                "ne2e-6484",
                "ne2e-6486",
                "ne2e-6488",
                "ne2e-6490",
                "ne2e-6491",
                "ne2e-6492",
                "ne2e-6493",
                "ne2e-6494",
                "ne2e-6495",
                "ne2e-6496",
                "ne2e-6497",
                "ne2e-6498",
                "ne2e-6499",
                "ne2e-6501",
                "ne2e-6500",
                "ne2e-6502",
                "ne2e-6503",
                "ne2e-6504",
                "ne2e-6505",
                "ne2e-6506",
                "ne2e-6507",
                "ne2e-6675",
                "ne2e-6782",
                "ne2e-6783",
                "ne2e-6784",
                "ne2e-6785",
                "ne2e-7234",
                "ne2e-7235",
                "ne2e-7236",
                "ne2e-7237",
                "ne2e-7238",
                "ne2e-7239",
                "ne2e-7260",
                "ne2e-7261",
                "ne2e-7262",
                "ne2e-7263",
                "ne2e-7264",
                "ne2e-7320",
                "ne2e-7325",
                "ne2e-7330",
                "ne2e-7372",
                "ne2e-7373",
                "ne2e-7420",
                "ne2e-7424",
                "ne2e-7428",
                "ne2e-7456",
                "ne2e-7457",
                "ne2e-7458",
                "ne2e-7470",
                "ne2e-7480",
                "ne2e-7507",
                "ne2e-7508",
                "ne2e-7509",
                "ne2e-7513",
                "ne2e-7739",
                "ne2e-7740",
                "ne2e-7741",
                "ne2e-7742",
                "ne2e-7743",
                "ne2e-8201",
                "ne2e-8310",
                "ne2e-8311",
                "ne2e-8312",
                "ne2e-8314",
                "ne2e-8315",
                "ne2e-8316",
                "ne2e-8317",
                "ne2e-8429",
                "ne2e-8430",
                "ne2e-8431",
                "ne2e-8432",
                "ne2e-8433",
                "ne2e-8434",
                "ne2e-8435",
                "ne2e-8436",
                "ne2e-8437",
                "ne2e-8438",
                "ne2e-8439",
                "ne2e-8440",
                "ne2e-8441",
                "ne2e-8442",
                "ne2e-8443",
                "ne2e-8446",
                "ne2e-8641",
                "ne2e-8900",
                "ne2e-8901",
                "ne2e-8902",
                "ne2e-8904",
                "ne2e-8938",
                "ne2e-8939",
                "ne2e-8940",
                "ne2e-8941",
                "ne2e-8942",
                "ne2e-8943",
                "ne2e-8944",
                "ne2e-8909",
                "ne2e-8910",
                "ne2e-8911",
                "ne2e-8913",
                "ne2e-8915",
                "ne2e-8917",
                "ne2e-8923",
                "ne2e-8924",
                "ne2e-8931",
                "ne2e-8932",
                "ne2e-8933",
                "ne2e-8934",
                "ne2e-8936",
                "ne2e-8937",
                "ne2e-8946",
                "ne2e-8947",
                "ne2e-8948",
                "ne2e-8951",
                "ne2e-8952",
                "ne2e-8953",
                "ne2e-8954",
                "ne2e-8955",
                "ne2e-8956",
                "ne2e-8959",
                "ne2e-8960",
                "ne2e-8962",
                "ne2e-8970",
                "ne2e-8963",
                "ne2e-8965",
                "ne2e-8966",
                "ne2e-8977",
                "ne2e-8979",
                "ne2e-8980",
                "ne2e-8971",
                "ne2e-8972",
                "ne2e-8973",
                "ne2e-8976",
                "ne2e-8982",
                "ne2e-8983",
                "ne2e-8985",
                "ne2e-8986",
                "ne2e-8987",
                "ne2e-8988",
                "ne2e-8991",
                "ne2e-8992",
                "ne2e-8993",
                "ne2e-8994",
                "ne2e-8995",
                "ne2e-8996",
                "ne2e-8997",
                "ne2e-8998",
                "ne2e-8999",
                "ne2e-9000",
                "ne2e-9001",
                "ne2e-9002",
                "ne2e-9003",
                "ne2e-9004",
                "ne2e-9005",
                "ne2e-9006",
                "ne2e-9008",
                "ne2e-9009",
                "ne2e-9010",
                "ne2e-9011",
                "ne2e-9012",
                "ne2e-9013",
                "ne2e-9014",
                "ne2e-9015",
                "ne2e-9016",
                "ne2e-9017",
                "ne2e-9018",
                "ne2e-9019",
                "ne2e-9020",
                "ne2e-9021",
                "ne2e-9022",
                "ne2e-9024",
                "ne2e-9025",
                "ne2e-9026",
                "ne2e-9029",
                "ne2e-9030",
                "ne2e-9031",
                "ne2e-9032",
                "ne2e-9033",
                "ne2e-9034",
                "ne2e-9036",
                "ne2e-9035",
                "ne2e-9037",
                "ne2e-9038",
                "ne2e-9039",
                "ne2e-9108",
                "ne2e-9109",
                "ne2e-9110",
                "ne2e-9111",
                "ne2e-9112",
                "ne2e-9113",
                "ne2e-9114",
                "ne2e-9115",
                "ne2e-9116",
                "ne2e-9117",
                "ne2e-9118",
                "ne2e-9119",
                "ne2e-9120",
                "ne2e-9121",
                "ne2e-9122",
                "ne2e-9123",
                "ne2e-9124",
                "ne2e-9125",
                "ne2e-9126",
                "ne2e-9127",
                "ne2e-9128",
                "ne2e-9129",
                "ne2e-9130",
                "ne2e-9131",
                "ne2e-9132",
                "ne2e-9133",
                "ne2e-9134",
                "ne2e-9135",
                "ne2e-9136",
                "ne2e-9137",
                "ne2e-9138",
                "ne2e-9141",
                "ne2e-9142",
                "ne2e-9143",
                "ne2e-9145",
                "ne2e-9146",
                "ne2e-11368",
                "ne2e-11369",
                "ne2e-11370",
                "ne2e-11371",
                "ne2e-11372",
                "ne2e-11373",
                "ne2e-11374",
                "ne2e-11375",
                "ne2e-11376",
                "ne2e-11654",
                "ne2e-11656",
                "ne2e-11657",
                "ne2e-11658",
                "ne2e-11659",
                "ne2e-11660",
                "ne2e-11661",
                "ne2e-11662",
                "ne2e-11663",
                "ne2e-11664",
                "ne2e-11665",
                "ne2e-11667",
                "ne2e-11668",
                "ne2e-11669",
                "ne2e-11670",
                "ne2e-11671",
                "ne2e-11672",
                "ne2e-11676",
                "ne2e-11674",
                "ne2e-11675",
                "ne2e-11677",
                "ne2e-11679",
                "ne2e-11680",
                "ne2e-11681",
                "ne2e-11682",
                "ne2e-11683",
                "ne2e-11685",
                "ne2e-11686",
                "ne2e-11687",
                "ne2e-11688",
                "ne2e-11690",
                "ne2e-11692",
                "ne2e-11691",
                "ne2e-12277",
                "ne2e-12278",
                "ne2e-12279",
                "ne2e-12280",
                "ne2e-12281",
                "ne2e-12326",
                "ne2e-12327",
                "ne2e-12332",
                "ne2e-12333",
                "ne2e-12328",
                "ne2e-12329",
                "ne2e-12330",
                "ne2e-12331",
                "ne2e-12334",
                "ne2e-12335",
                "ne2e-12336",
                "ne2e-12338",
                "ne2e-12339",
                "ne2e-12340",
                "ne2e-12506",
                "ne2e-12507",
                "ne2e-12510",
                "ne2e-12511",
                "ne2e-12512",
                "ne2e-12513",
                "ne2e-12514",
                "ne2e-3143",
                "ne2e-3169",
                "ne2e-3144",
                "ne2e-3145",
                "ne2e-3146",
                "ne2e-3173",
                "ne2e-3174",
                "ne2e-3147",
                "ne2e-3148",
                "ne2e-3177",
                "ne2e-3178",
                "ne2e-3149",
                "ne2e-3150",
                "ne2e-3151",
                "ne2e-3152",
                "ne2e-3153",
                "ne2e-3154",
                "ne2e-3155",
                "ne2e-3156",
                "ne2e-3157",
                "ne2e-3159",
                "ne2e-3164",
                "ne2e-3168",
                "ne2e-3256",
                "ne2e-3160",
                "ne2e-3161",
                "ne2e-3162",
                "ne2e-3253",
                "ne2e-3254",
                "ne2e-3204",
                "ne2e-3205",
                "ne2e-3206",
                "ne2e-3207",
                "ne2e-3208",
                "ne2e-3212",
                "ne2e-3216",
                "ne2e-3215",
                "ne2e-3217",
                "ne2e-3218",
                "ne2e-3219",
                "ne2e-3220",
                "ne2e-3222",
                "ne2e-3223",
                "ne2e-3221",
                "ne2e-3224",
                "ne2e-3225",
                "ne2e-3226",
                "ne2e-3227",
                "ne2e-3239",
                "ne2e-3229",
                "ne2e-3230",
                "ne2e-3231",
                "ne2e-3242",
                "ne2e-3243",
                "ne2e-3234",
                "ne2e-3233",
                "ne2e-3235",
                "ne2e-3236",
                "ne2e-3237",
                "ne2e-3238",
                "ne2e-3240",
                "ne2e-3241",
                "ne2e-3244",
                "ne2e-3245",
                "ne2e-3251",
                "ne2e-3252",
                "ne2e-3249",
                "ne2e-3248",
                "ne2e-3250",
                "ne2e-3257",
                "ne2e-3258",
                "ne2e-3259",
                "ne2e-3260",
                "ne2e-3261",
                "ne2e-3262",
            ]:
                new_rq1_eval.append(dt)
        print(f"Sanity check passed for rq1-eval")

        save_dataset(Macros.data_dir / "rq1-eval", new_rq1_eval)


def find_netests_in_same_class(
    test_methods: List[TestMethod], etest_key: str
) -> List[int]:
    """
    Given etest key, find the tests in the same class.
    """

    # find etest item
    etest_method = [
        tm
        for tm in test_methods
        if tm.cname.replace(".", "/") + "#" + tm.mname + "#()V" == etest_key
    ]

    etest_class_name = etest_method[0].cname
    # find the tests in the same class
    netest_id = [
        i
        for i, tm in enumerate(test_methods)
        if tm.cname == etest_class_name and tm.pattern is None
    ]

    return netest_id


def collect_maven_module_methods(maven_module: MavenModule):
    """
    Collect all the methods in the target maven module.
    """
    temp_dir = su.io.mktmp_dir("etestgen")
    # collect all source code in the project
    su.io.dump(
        temp_dir / "config.json",
        {
            "mainSrcRoot": maven_module.main_srcpath,
            "classpath": maven_module.dependency_classpath,
            "testSrcRoot": maven_module.test_srcpath,
            "outPath": str(temp_dir / "out.json"),
            "debugPath": str(temp_dir / "debug.txt"),
        },
    )
    su.bash.run(
        f"java -cp {Tool.core_jar} org.etestgen.core.SrcMainMethodCollector {temp_dir}/config.json",
        0,
    )
    records_methods = su.io.load(
        temp_dir / "out.json"
    )  # method name and method body (tokens)
    su.io.rmdir(temp_dir)
    return records_methods


def extract_module_test_coverage(mut2e_dt: TData, coverage_out_dir: Union[Path, str]):
    """
    Prepare coverage data for the target module.
    """
    cur_prj = mut2e_dt.project
    cur_module = mut2e_dt.module_i
    module_cov_dir = coverage_out_dir / cur_prj / str(cur_module)
    test_methods = su.io.load(
        module_cov_dir / "manual.tests.jsonl",
        clz=TestMethod,
    )
    try:
        test2mut = su.io.load(module_cov_dir / "netest-coverage.jsonl")
    except FileNotFoundError:
        logger.warning(f"Can not find coverage data for {cur_prj}.{cur_module}")
        return None, test_methods
    mut2test = defaultdict(list)
    for dt in test2mut:
        test_i = dt["test_i"]
        for mut_key in dt["methods"]:
            mut2test[mut_key].append(int(test_i))

    return mut2test, test_methods


def prepare_module_data(mut2e_dt: TData):
    """
    Prepare the necessary data for ne test collection.
    """

    coverage_out_dir = Macros.work_dir / "coverage"
    cur_prj = mut2e_dt.project
    cur_module = mut2e_dt.module_i
    module_cov_dir = coverage_out_dir / cur_prj / str(cur_module)
    test_methods = su.io.load(
        module_cov_dir / "manual.tests.jsonl",
        clz=TestMethod,
    )
    try:
        test2mut = su.io.load(module_cov_dir / "test_2_mut.json")
    except FileNotFoundError:
        return None, test_methods

    return test2mut, test_methods


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.INFO)
    CLI(DataCollector, as_positional=False)
