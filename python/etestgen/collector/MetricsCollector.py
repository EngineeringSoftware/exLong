import os
from collections import defaultdict
import seutil as su
from jsonargparse import CLI
from typing import List, Dict, Any, Union, Tuple
from statistics import mean, median
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from seutil.maven import MavenModule, MavenProject
from seutil.project import Project

from etestgen.data.tool import Tool
from etestgen.macros import Macros
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.data.data import DataNE2E, TData, parse_data_cls
from etestgen.eval.compute_throws_coverage import TestMethod
from etestgen.eval.extract_etest_data_from_coverage import collect_module_source_code

logger = su.log.get_logger(__name__, su.LoggingUtils.INFO)


class MetricsCollector:
    def __init__(self) -> None:
        self.metrics_dir = Macros.results_dir / "stats"
        su.io.mkdir(self.metrics_dir)

    def collect_project_level_stats(self):
        """
        Collect the following numbers for projects:
        - total number of methods (method_record)
        - total number of tests
        - total number of classes
        """

        project_list = su.io.load(self.metrics_dir / "mut2e-projects-list.json")
        project_level_stats = {}
        for project in tqdm(project_list, total=len(project_list)):
            project_stats_dir = Macros.work_dir / "coverage-new" / project
            project_test_count = 0
            project_method_count = 0
            project_class_count = 0
            project_etype_count = 0
            java_loc = get_project_cloc(Macros.downloads_dir / project, project)
            for sub_dir in project_stats_dir.iterdir():
                if sub_dir.is_dir():
                    try:
                        method_lists = su.io.load(sub_dir / "method_records.jsonl")
                        class_name_list = set(
                            [m["method"].split("#")[0] for m in method_lists]
                        )
                        project_class_count += len(class_name_list)
                        project_method_count += len(method_lists)
                        test_lists = su.io.load(
                            sub_dir / "manual.tests.jsonl", clz=TestMethod
                        )
                        project_test_count += len(test_lists)
                        etype_list = [
                            t.exception for t in test_lists if t.exception is not None
                        ]
                        project_etype_count += len(set(etype_list))
                    except FileNotFoundError:
                        logger.warning(f"Cannot find the target files in {sub_dir}")
            project_level_stats[project] = {
                "test-method-count": project_test_count,
                "nontest-method-count": project_method_count,
                "nontest-class-count": project_class_count,
                "loc-count": java_loc,
                "etype-count": project_etype_count,
            }
        #
        su.io.dump(
            self.metrics_dir / "stats-project-level.json",
            project_level_stats,
            su.io.Fmt.jsonPretty,
        )

    def collect_stats_for_eval_set(self, which: str):
        """
        collect: ebts,mut,throw,exception_type
        """
        # load the dataset
        rq1_eval_dataset = load_dataset(Macros.data_dir / which, clz=DataNE2E)

        # collect stats
        project_set = set()
        exception_set = set()
        module_set = set()
        mut_set = set()
        ts_set = set()

        for dt in rq1_eval_dataset:
            project_set.add(dt.project)
            exception_set.add(dt.etype)
            module_name = dt.project + "#" + dt.module
            module_set.add(module_name)
            mut_set.add(dt.mut_key)
            ts = dt.e_stack_trace[-1][0]["method"] + str(dt.e_stack_trace[-1][1])
            ts_set.add(ts)

        stat = {
            "count-project": len(project_set),
            "count-etype": len(exception_set),
            "count-module": len(module_set),
            "count-mut": len(mut_set),
            "count-ts": len(ts_set),
            "ebts": len(rq1_eval_dataset),
        }
        su.io.dump(self.metrics_dir / f"stats-{which}.json", stat, su.io.Fmt.jsonPretty)

    def collect_stats_for_rq2(
        self,           
        repos_file: Path = Macros.work_dir / "repos" / "filtered" / "repos.json",
        test_projects_list_file: Path = Macros.results_dir / "repos" / "test-projects.json",
        coverage_dir: Path =  Macros.work_dir / "coverage-new",
        data_dir: Path = Macros.data_dir,
    ):
        """
        - # total number of throw statements
        - mean # throw statements per methods (if > 0)
        - # total number of methods
        - among methods, # are public methods
        """
        Tool.ensure_tool_versions()
        Tool.require_compiled()

        def skip_module(maven_module: Any, module_i: int):
            skipped = None
            if maven_module.packaging == "pom":
                # skip parent/aggregator modules
                skipped = (module_i, maven_module.coordinate, "package==pom")
                return skipped
            if (
                not Path(maven_module.main_srcpath).exists()
                or not Path(maven_module.test_srcpath).exists()
            ):
                # skip non-standard modules
                skipped = (module_i, maven_module.coordinate, "missing src")
                return skipped
            return skipped

        test_projects_list = su.io.load(test_projects_list_file)
        projects = su.io.load(repos_file, clz=List[Project])
        indexed_projects: Dict[str, Project] = {p.full_name: p for p in projects}
        count_methods_with_ts = 0
        count_methods_with_ts_public = 0
        count_ts = 0
        count_public_ts = 0
        rq2_dataset, rq2_public_methods = [], []
        method_ts = []
        index = 0

        for prj_name in test_projects_list:
            project_dir = coverage_dir / prj_name
            project = indexed_projects[prj_name]
            project.clone(Macros.downloads_dir / prj_name)

            for module_dir in project_dir.iterdir():
                if not module_dir.is_dir():
                    continue
                maven_module = su.io.load(module_dir / "module.yaml", clz=MavenModule)
                maven_module.project = project
                module_i = int(os.path.basename(module_dir))
                if skip_module(maven_module, module_i):
                    continue
                try:
                    method_records = su.io.load(module_dir / "method_records.jsonl")
                except FileNotFoundError:
                    method_records = collect_module_source_code(
                        indexed_projects[prj_name],
                        Macros.work_dir / "downloads",
                        maven_module.rel_path,
                    )
                    su.io.dump(module_dir / "method_records.jsonl", method_records)

                for m in method_records:
                    if not m["exceptions"]:
                        continue
                    count_methods_with_ts += 1
                    if "protected" in m["modifiers"] or "private" in m["modifiers"]:
                        for ex in m["exceptions"]:
                            dt = DataNE2E()
                            dt.id = index
                            dt.project = prj_name
                            dt.mut_key = m["method"]
                            dt.mut = m["method_node"]
                            dt.module_i = module_i
                            dt.module = maven_module.rel_path
                            ex_line = int(ex.split("@@")[1]) - int(m["startLine"])
                            dt.e_stack_trace = [[m, ex_line]]
                            dt.etype = ex.split("@@")[0]
                            rq2_dataset.append(dt)
                            index += 1
                    else:
                        count_methods_with_ts_public += 1
                        for ex in m["exceptions"]:
                            dt = DataNE2E()
                            dt.id = index
                            dt.project = prj_name
                            dt.module_i = module_i
                            dt.module = maven_module
                            dt.module = maven_module.rel_path
                            dt.mut_key = m["method"]
                            dt.mut = m["method_node"]
                            ex_line = int(ex.split("@@")[1]) - int(m["startLine"])
                            dt.e_stack_trace = [[m, ex_line]]
                            dt.etype = ex.split("@@")[0]
                            rq2_dataset.append(dt)
                            rq2_public_methods.append(dt)
                            index += 1
                        count_public_ts += len(m["exceptions"])
                    count_ts += len(m["exceptions"])
                    method_ts.append(len(m["exceptions"]))
        #
        stats = {
            "count-methods-with-ts": count_methods_with_ts,
            "count-methods-with-ts-public": count_methods_with_ts_public,
            "count-ts": count_ts,
            "count-public-ts": count_public_ts,
            "mean-ts-per-method": mean(method_ts),
        }
        save_dataset(data_dir / "rq2-all", rq2_dataset, clz=DataNE2E)
        save_dataset(data_dir / "rq2-public", rq2_public_methods, clz=DataNE2E)
        su.io.dump(
            self.metrics_dir / "stats-methods-with-ts.json", stats, su.io.Fmt.jsonPretty
        )

    def collect_ts_rq2_per_project(self):
        """
        Collect the number of throw statements per project.
        """

        dataset_names = ["rq2-all", "rq2-public", "rq2"]
        for dt_name in dataset_names:
            rq2_dataset = load_dataset(Macros.data_dir / dt_name, clz=DataNE2E)
            project_ts = defaultdict(int)
            for dt in rq2_dataset:
                project_ts[dt.project] += 1
            su.io.dump(
                self.metrics_dir / f"{dt_name}-project-ts.json",
                project_ts,
                su.io.Fmt.jsonPretty,
            )

    def collect_mut_etype_stats(self):
        """
        1. # stack trace of length 1
        2. mut_with_stack_trace_1, (set of exceptions)
        3. mut, set(etype)
        """
        from collections import Counter

        data_dir = Macros.data_dir / "mut2e"
        data_cls = parse_data_cls("MUT2E")
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(data_dir, clz=data_cls, pbar=pbar)
        # Collect stats
        mut_more_than_one_etype = 0
        one_stack_trace_count = 0
        one_stack_trace_with_throw_stmt = 0
        test_throw_stmt_count = 0
        tested_etype_dist = []
        mut_throw_stmt_dist = []
        mut_no_throw_stmt_count = 0

        throw_stmt_etypes_list = []
        third_party_ids = []
        third_party_etypes_list = []
        mut_2_etype = defaultdict(set)
        for dt in mut2e_dataset:
            if len(dt.e_stack_trace) == 1:
                one_stack_trace_count += 1
                throw_line_number = dt.e_stack_trace[-1][1]
                try:
                    thrown_line = dt.e_stack_trace[-1][0]["method_node"].splitlines()[
                        throw_line_number
                    ]
                    if not (not "throw " in thrown_line and ";" in thrown_line):
                        one_stack_trace_with_throw_stmt += 1
                except IndexError:
                    pass
            #
            throw_line_number = dt.e_stack_trace[-1][1]
            try:
                thrown_line = dt.e_stack_trace[-1][0]["method_node"].splitlines()[
                    throw_line_number
                ]
                if not (not "throw " in thrown_line and ";" in thrown_line):
                    test_throw_stmt_count += 1
                    throw_stmt_etypes_list.append(dt.etype)
                else:
                    third_party_etypes_list.append(dt.etype)
                    third_party_ids.append(dt.id)
            except IndexError:
                third_party_etypes_list.append(dt.etype)
                pass
            #
            mut_2_etype[dt.mut_key].add(dt.etype)
            mut_body_lines = dt.mut.splitlines()
            throw_stmt_count = 0
            for line in mut_body_lines:
                if "throw " in line:
                    throw_stmt_count += 1
            mut_throw_stmt_dist.append(throw_stmt_count)
            if throw_stmt_count == 0:
                mut_no_throw_stmt_count += 1
        #
        for _, v in mut_2_etype.items():
            tested_etype_dist.append(len(v))
        for etype_count in tested_etype_dist:
            if etype_count > 1:
                mut_more_than_one_etype += 1

        throw_stmt_etype_counter = Counter(throw_stmt_etypes_list)
        third_party_etype_counter = Counter(third_party_etypes_list)
        print(f"=== Most common throw stmt etypes ===")
        print(throw_stmt_etype_counter.most_common(10))
        print(f"=== Most common third party etypes ===")
        print(third_party_etype_counter.most_common(10))
        print(third_party_ids)
        stats = {
            "max-mut-throw-stmts": max(mut_throw_stmt_dist),
            "min-mut-throw-stmts": min(mut_throw_stmt_dist),
            "median-mut-throw-stmts": median(mut_throw_stmt_dist),
            "mut-no-throw-stmt": mut_no_throw_stmt_count,
            "max-mut-etypes": max(tested_etype_dist),
            "median-mut-etypes": median(tested_etype_dist),
            "more-than-one-etype": mut_more_than_one_etype,
            "stack-trace-len-1": one_stack_trace_count,
            "stack-trace-len-1-with-throw-stmt": one_stack_trace_with_throw_stmt,
            "test-throw-stmt-count": test_throw_stmt_count,
        }
        su.io.dump(
            self.metrics_dir / "stats-mut-etype.json", stats, su.io.Fmt.jsonPretty
        )

    def collect_mut2e_stats(self):
        """
        Collect stats for MUT2E dataset and write to metrics_dir.
        """

        data_dir = Macros.data_dir / "ne2e"
        data_cls = parse_data_cls("NE2E")
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(data_dir, clz=data_cls, pbar=pbar)
        # Collect stats
        stack_trace_outlier = 0
        stats = {}
        exceptions_set = set()
        mut_set = set()
        project_set = set()
        pattern_stats = defaultdict(int)
        netest_dist = []
        no_call_trace = 0
        modules_set = set()

        new_mut2e_dataset = []
        for dt in mut2e_dataset:
            module_name = dt.project + "#" + dt.module
            modules_set.add(module_name)
            pattern_stats[dt.etest_pattern] += 1
            if len(dt.e_stack_trace) > 6:
                # process the case where the stack trace is too long
                dt = update_dt_remove_duplicate_stack_trace(dt)
            depth = len(dt.e_stack_trace)
            if depth > 6:
                stack_trace_outlier += 1
            netest_dist.append(depth)
            mut_set.add(dt.mut_key)
            project_set.add(dt.project)
            exceptions_set.add(dt.etype)
            new_mut2e_dataset.append(dt)
        #
        stats["mut-count"] = len(mut_set)
        stats["pattern-stats"] = pattern_stats
        stats["project-count"] = len(project_set)
        stats["exception-count"] = len(exceptions_set)
        stats["avg-stack-trace-len"] = mean(netest_dist)
        stats["no-stack-trace-count"] = no_call_trace
        stats["median-stack-trace-len"] = median(netest_dist)
        stats["max-stack-trace-len"] = max(netest_dist)
        stats["stack-trace-len>6"] = stack_trace_outlier
        stats["min-stack-trace-len"] = min(netest_dist)
        stats["etest-count"] = len(mut2e_dataset)
        stats["module-count"] = len(modules_set)
        # box plot for call stack depth
        plt.boxplot(netest_dist)
        plt.xlabel("Call Stack Depth")
        plt.ylabel("# method (excluding the test method)")
        plt.title("Call Stack Depth Distribution")
        plt.savefig(self.metrics_dir / "call-stack-depth.png")
        su.io.dump(
            self.metrics_dir / "stats-mut2e-dataset.json", stats, su.io.Fmt.jsonPretty
        )
        su.io.dump(self.metrics_dir / "mut2e-projects-list.json", list(project_set))
        # save_dataset(Macros.data_dir / "mut2e", new_mut2e_dataset)

    def collect_dataset_split_stats(self):
        data_dir = Macros.data_dir / "mut2e"
        data_cls = parse_data_cls("MUT2E")
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(data_dir, clz=data_cls, pbar=pbar)
        total_project_set = set([d.project for d in mut2e_dataset])

        projects_split_stats = {
            "train": {
                "test-count": 0,
                "module-count": 0,
                "etest-count": 0,
                "project-count": 0,
                "etype-count": 0,
                "etypes": [],
                "mut-count": 0,
                "mut": [],
                "modules": [],
            },
            "valid": {
                "test-count": 0,
                "etest-count": 0,
                "module-count": 0,
                "project-count": 0,
                "etype-count": 0,
                "etypes": [],
                "mut-count": 0,
                "mut": [],
                "modules": [],
            },
            "test": {
                "test-count": 0,
                "etest-count": 0,
                "module-count": 0,
                "project-count": 0,
                "etype-count": 0,
                "etypes": [],
                "mut-count": 0,
                "mut": [],
                "modules": [],
            },
        }
        for split in ["train", "valid", "test"]:
            split_projects = su.io.load(
                Macros.results_dir / "repos" / f"{split}-projects.json"
            )
            projects_split_stats[split]["projects"] = list(
                set(split_projects).intersection(total_project_set)
            )
            # load the mut2e dataset
            projects_split_stats[split]["project-count"] = len(
                projects_split_stats[split]["projects"]
            )
            su.io.dump(
                Macros.results_dir / "repos" / f"{split}-projects.json",
                projects_split_stats[split]["projects"],
            )
            # test for each split
            split_test_count = 0
            project_list = projects_split_stats[split]["projects"]
            for prj in project_list:
                coverage_data_dir = Macros.work_dir / "coverage-new" / prj
                subdirectories = [
                    os.path.join(coverage_data_dir, d)
                    for d in os.listdir(coverage_data_dir)
                    if os.path.isdir(os.path.join(coverage_data_dir, d))
                ]
                for sub_dir in subdirectories:
                    if os.path.exists(Path(sub_dir) / "manual.tests.jsonl"):
                        manual_tests = su.io.load(
                            Path(sub_dir) / "manual.tests.jsonl", clz=List[TestMethod]
                        )
                        split_test_count += len(manual_tests)
                #
            #
            projects_split_stats[split]["test-count"] = split_test_count

        for split in ["train", "valid", "test"]:
            for dt in mut2e_dataset:
                if dt.project in projects_split_stats[split]["projects"]:
                    projects_split_stats[split]["etest-count"] += 1
                    projects_split_stats[split]["modules"].append(
                        f"{dt.project}#{dt.module}"
                    )
                    projects_split_stats[split]["etypes"].append(dt.etype)
                    projects_split_stats[split]["mut"].append(dt.mut_key)
        #
        for split in ["train", "valid", "test"]:
            projects_split_stats[split]["etype-count"] = len(
                set(projects_split_stats[split]["etypes"])
            )
            projects_split_stats[split]["module-count"] = len(
                set(projects_split_stats[split]["modules"])
            )
            del projects_split_stats[split]["modules"]
            del projects_split_stats[split]["etypes"]
            projects_split_stats[split]["mut-count"] = len(
                set(projects_split_stats[split]["mut"])
            )
            del projects_split_stats[split]["mut"]
            del projects_split_stats[split]["projects"]

        #
        su.io.dump(
            self.metrics_dir / "stats-dataset-projects-split.json",
            projects_split_stats,
            su.io.Fmt.jsonPretty,
        )

    def collect_all_tests(self):
        projects = su.io.load(self.metrics_dir / "mut2e-projects-list.json")
        test_count = 0
        for prj in tqdm(projects, total=len(projects)):
            coverage_data_dir = Macros.work_dir / "coverage-new" / prj
            subdirectories = [
                os.path.join(coverage_data_dir, d)
                for d in os.listdir(coverage_data_dir)
                if os.path.isdir(os.path.join(coverage_data_dir, d))
            ]
            for sub_dir in subdirectories:
                manual_tests = su.io.load(
                    Path(sub_dir) / "manual.tests.jsonl", clz=List[TestMethod]
                )
                test_count += len(manual_tests)
            #
        #
        data_stats = su.io.load(self.metrics_dir / "stats-mut2e-dataset.json")
        data_stats["test-count"] = test_count
        su.io.dump(
            self.metrics_dir / "stats-mut2e-dataset.json",
            data_stats,
            su.io.Fmt.jsonPretty,
        )

    def collect_ne2e_stats(self):
        """
        Collect stats for NE2E dataset and write to metrics_dir.
        """
        stats = {}

        data_dir = Macros.data_dir / "ne2e"
        data_cls = parse_data_cls("NE2E")
        with tqdm("Loading NE2E data") as pbar:
            ne2e_dataset = load_dataset(data_dir, clz=data_cls, pbar=pbar)
        e2ne = []
        with tqdm(total=len(ne2e_dataset), desc="Iterating ne2e data") as pbar:
            for nedt in ne2e_dataset:
                e2ne.append(len(nedt.test_ne_key))
                pbar.update(1)
        #
        stats["avg-netest-per-etest"] = mean(e2ne)
        stats["median-netest-per-etest"] = median(e2ne)
        stats["max-netest-per-etest"] = max(e2ne)
        stats["min-netest-per-etest"] = min(e2ne)
        stats["count-etest-no-netest"] = e2ne.count(0)
        stats["count-ne2e-pairs"] = sum(e2ne)

        su.io.dump(
            self.metrics_dir / "stats-ne2e-dataset.json", stats, su.io.Fmt.jsonPretty
        )

    def collect_ne2e_test_data_with_st(
            self, 
            ne2e_data_path: Path = Macros.work_dir / "setup" 
                                                / "ne2e-with-name-ft" 
                                                / "eval" 
                                                / "test",
            out_ids_file: str = "test-set-has-throw-statement-ids.json",
            out_projects_file: str = "test-set-has-throw-statement-projects.json",
            
        ):
        """
        Collect the data ids where the data has throw statement as the last called line.
        """
        ne2e_ground_truth_test_data = load_dataset(
            ne2e_data_path, clz=DataNE2E,
        )
        throw_statement_list, no_throw_statement_list = [], []
        project_set = set()
        for dt in ne2e_ground_truth_test_data:
            stack_trace = dt.e_stack_trace
            thrown_exceptions = [e[:e.find("@@")] for e in stack_trace[-1][0]["exceptions"]]
            if (
                dt.etype in thrown_exceptions
                or dt.etype.split(".")[-1] in thrown_exceptions
            ):
                throw_statement_list.append(dt.id)
                project_set.add(dt.project)
            else:
                no_throw_statement_list.append(dt.id)
        logger.info(
            f"Number of test dataset that the exception is thrown from throw statement: {len(throw_statement_list)}"
        )
        logger.info(f"Number of projects for inference: {len(project_set)}")
        su.io.dump(
            self.metrics_dir / out_ids_file,
            throw_statement_list,
        )
        su.io.dump(
            self.metrics_dir / out_projects_file,
            list(project_set),
        )

    def collect_eval_stack_trace_metrics(self):
        """
        # projects
        # etype from throw statement
        # static analysis helps
        # cases where more than one real stack traces are collected.
        """
        count_eval_throw_stmt_data = 487
        more_than_one_real_stack_trace = 0
        project_set = set()

        test_data = load_dataset(
            Macros.data_dir / "ne2e-test-st-with-line-number", clz=DataNE2E
        )
        count_static_analysis_helps = len(test_data)
        for dt in test_data:
            project_set.add(dt.project)
            if len(dt.e_stack_trace) > 1:
                more_than_one_real_stack_trace += 1
        #
        stats = {
            "count-eval-throw-stmt-data": count_eval_throw_stmt_data,
            "count-static-analysis-helps": count_static_analysis_helps,
            "count-projects": len(project_set),
            "count-st-more-than-one": more_than_one_real_stack_trace,
        }
        static_tool_help_ids = [d.id for d in test_data]
        su.io.dump(
            self.metrics_dir / "stats-eval-stack-trace.json",
            stats,
        )
        su.io.dump(
            self.metrics_dir / "eval-stack-trace-static-tool-help-ids.json",
            static_tool_help_ids,
        )

    def collect_test_stack_trace_metrics(self, coverage_dir: Path):
        """
        Collect stats on ground-truth stack traces in test set.
        # data
        # last called line is throw statement
        # last called line is a method call (runtime exception)
        # pct that we are able to collect from netest
        """
        real_ne2e_test_data = load_dataset(Macros.data_dir / "ne2e-test", clz=DataNE2E)
        with_stack_trace_data = []
        without_stack_trace_data = []
        for dt in real_ne2e_test_data:
            etype = dt.etype.split(".")[-1]
            if len(dt.e_stack_trace) > 0:
                with_stack_trace_data.append(dt.id)
            else:
                # check if the throw statement is in the mut
                # load method record
                method_records = su.io.load(
                    coverage_dir
                    / dt.project
                    / str(dt.module_i)
                    / "method_records.jsonl"
                )
                method2code = {}
                for record in method_records:
                    ms_key = record["method"]
                    method2code[ms_key] = record
                if (
                    dt.etype in method2code[dt.mut_key]["exceptions"]
                    or etype in method2code[dt.mut_key]["exceptions"]
                ):
                    with_stack_trace_data.append(dt.id)
                else:
                    without_stack_trace_data.append(dt.id)

        ne2e_ground_truth_test_data = load_dataset(
            Macros.work_dir / "setup" / "ne2e-with-name-ft" / "eval" / "test",
            clz=DataNE2E,
        )
        throw_statement_list = []
        no_throw_statement_list = []
        for dt in ne2e_ground_truth_test_data:
            if dt.id not in without_stack_trace_data:
                continue
            stack_trace = dt.e_stack_trace
            thrown_exceptions = stack_trace[-1][0]["exceptions"]
            if (
                dt.etype in thrown_exceptions
                or dt.etype.split(".")[-1] in thrown_exceptions
            ):
                throw_statement_list.append(dt.id)
            else:
                no_throw_statement_list.append(dt.id)
        #
        stats = {
            "collected-stack-trace": len(with_stack_trace_data),
            "no-stack-trace-throw-statement": len(throw_statement_list),
            "no-stack-trace-method-call": len(no_throw_statement_list),
        }
        su.io.dump(
            self.metrics_dir / "stats-test-set-stack-trace.json",
            stats,
            su.io.Fmt.jsonPretty,
        )
        su.io.dump(
            self.metrics_dir / "test-set-no-stack-trace-ids.json",
            without_stack_trace_data,
        )

    def collect_test_data_stack_trace_stats(self):
        """
        Collect stats for the stack traces in the test data.
        """

        data_dir = Macros.data_dir / "ne2e-test"
        data_cls = parse_data_cls("NE2E")
        no_stack_trace_ids = su.io.load(
            self.metrics_dir / "test-set-no-stack-trace-ids.json"
        )
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(data_dir, clz=data_cls, pbar=pbar)
        # Collect stats
        stats = {}
        stack_trace_counter = []
        stack_trace_depth = []
        no_stack_trace = 0
        # netest
        netest_counter = []
        no_netest = 0

        condition_counter = []
        no_condition = 0

        for dt in mut2e_dataset:
            if len(dt.test_ne_key) == 0:
                no_netest += 1
            if dt.id in no_stack_trace_ids:
                no_stack_trace += 1
                assert len(dt.condition) == 0
                no_condition += 1
                continue
            netest_counter.append(len(dt.test_ne_key))

            cleaned_stack_trace = clean_all_stack_trace(dt.e_stack_trace)
            stack_trace_counter.append(len(cleaned_stack_trace))
            for stack_trace in cleaned_stack_trace:
                if len(stack_trace) > 20:
                    stack_trace = remove_duplicate_stack_trace(stack_trace)
                depth = max(len(stack_trace), 1)
                stack_trace_depth.append(depth)

            if len(dt.condition) == 0:
                no_condition += 1
            else:
                condition_counter.append(len(set(dt.condition)))
        #
        stats["avg-stack-trace-len"] = mean(stack_trace_depth)
        stats["median-stack-trace-len"] = median(stack_trace_depth)
        stats["max-stack-trace-len"] = max(stack_trace_depth)
        stats["min-stack-trace-len"] = min(stack_trace_depth)

        stats["no-stack-trace-count"] = no_stack_trace
        stats["with-stack-trace-count"] = len(mut2e_dataset) - no_stack_trace
        stats["avg-stack-trace-count"] = mean(stack_trace_counter)
        stats["max-stack-trace-count"] = max(stack_trace_counter)
        stats["median-stack-trace-count"] = median(stack_trace_counter)

        stats["no-netest-count"] = no_netest
        stats["avg-netest-count"] = mean(netest_counter)
        stats["with-netest-count"] = len(mut2e_dataset) - no_netest
        stats["median-netest-count"] = median(netest_counter)
        stats["max-netest-count"] = max(netest_counter)

        stats["no-condition-count"] = no_condition
        stats["avg-condition-count"] = mean(condition_counter)

        su.io.dump(
            self.metrics_dir / "stats-test-dataset.json",
            stats,
            su.io.Fmt.jsonPretty,
        )

    def collect_stack_trace_and_condition_stats(self):
        """
        Collect the stats for stack traces and conditions in test set.
        """

        data_dir = Macros.data_dir / "ne2e-test"
        ne2e_test_dataset = load_dataset(data_dir, clz=DataNE2E)

        no_stack_trace = 0
        num_stack_traces = []
        num_conditions = []
        for dt in tqdm(ne2e_test_dataset):
            if len(dt.e_stack_trace) == 0 and dt.etype.split(".")[-1] not in dt.mut:
                no_stack_trace += 1
            else:
                num_stack_trace = len(dt.e_stack_trace)
                num_condition = len(
                    list(set([cond for cond in dt.condition if cond != ""]))
                )
                if num_condition > 1 and num_condition <= 5:
                    print(dt.id)
                num_stack_traces.append(num_stack_trace)
                num_conditions.append(num_condition)
        #
        stats = {
            "no-stack-trace": no_stack_trace,
            "avg-num-stack-traces": mean(num_stack_traces),
            "median-num-stack-trace": median(num_stack_traces),
            "avg-num-conditions": mean(num_conditions),
            "median-num-conditions": median(num_conditions),
        }
        su.io.dump(self.metrics_dir / "stats-eval-data-conditions.json", stats)

    def collect_ground_truth_stack_traces_stats(self):
        """
        Collect the stats for ground truth stacks in test set.
        """
        data_dir = Macros.data_dir / "ne2e"
        ne2e_dataset = load_dataset(data_dir, clz=DataNE2E)
        test_projects_list = su.io.load(
            Macros.results_dir / "repos" / "test-projects.json"
        )

        correct_throw_line = 0
        try_catch_block = 0
        gt_good_stack_trace_set = set()
        for dt in tqdm(ne2e_dataset):
            if dt.project not in test_projects_list:
                continue
            e_stack_trace = dt.e_stack_trace
            last_called_method = e_stack_trace[-1][0]["method_node"]
            last_called_line = e_stack_trace[-1][1]
            throw_line = last_called_method.splitlines()[last_called_line]
            if "throw" in throw_line:
                correct_throw_line += 1
                gt_good_stack_trace_set.add(dt.id)
            elif dt.etype.split(".")[-1] in last_called_method:
                try_catch_block += 1
                gt_good_stack_trace_set.add(dt.id)
        #
        ne2e_test = load_dataset(Macros.data_dir / "ne2e-test", clz=DataNE2E)
        real_good_stack_trace_set = set()
        more_than_one_real_stack_trace = 0
        for dt in tqdm(ne2e_test):
            if len(dt.e_stack_trace) > 0:
                more_than_one_real_stack_trace += 1
                real_good_stack_trace_set.add(dt.id)
        #
        print(
            f"{correct_throw_line} ground-truth stack traces have the throw statement. {try_catch_block} ground-truth stack traces have the try-catch block."
        )
        print(
            f"When collect stack trace from netest, {more_than_one_real_stack_trace} netests have more than one real stack trace."
        )
        union_good_data = gt_good_stack_trace_set.union(real_good_stack_trace_set)
        diff_good_data = gt_good_stack_trace_set.difference(real_good_stack_trace_set)
        print(f"Union good data size: {len(union_good_data)}")
        print(f"Diff good data size: {len(diff_good_data)}")

    def make_barplot(
        self, data: Union[dict, Tuple], xlabel: str, ylabel: str, title: str
    ):
        item_names = [dt[0] for dt in data]
        item_values = [dt[1] for dt in data]
        plt.bar(item_names, item_values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.savefig(self.metrics_dir / f"{title}-barplot.png")


def get_project_cloc(project_dir: Path, project_name: str):
    with su.io.cd(project_dir):
        project_stats_dir = Macros.work_dir / "coverage-new" / project_name
        su.bash.run(
            f"cloc . --yaml --out={project_stats_dir}/{project_name}-cloc.yaml", 0
        )
        project_cloc = su.io.load(f"{project_stats_dir}/{project_name}-cloc.yaml")
        loc = project_cloc["Java"]["code"]
    return loc


def update_dt_remove_duplicate_stack_trace(dt: Any):
    exist_cm = set()

    new_stack_trace = []
    for st in dt.e_stack_trace[:-1]:
        if str(st) not in exist_cm:
            exist_cm.add(str(st))
            new_stack_trace.append(st)
    new_stack_trace.append(dt.e_stack_trace[-1])
    dt.e_stack_trace = new_stack_trace
    return dt


def clean_all_stack_trace(stack_trace_list: List):
    """Clean the list of stack traces if there are duplicates."""
    stack_trace_set = set()
    new_stack_trace_list = []
    for st in stack_trace_list:
        if str(st) not in stack_trace_set:
            stack_trace_set.add(str(st))
            new_stack_trace_list.append(st)
    return new_stack_trace_list


def remove_duplicate_stack_trace(stack_trace: List):
    exist_cm = set()

    new_stack_trace = []
    for st in stack_trace[:-1]:
        if str(st) not in exist_cm:
            exist_cm.add(str(st))
            new_stack_trace.append(st)
    new_stack_trace.append(stack_trace[-1])
    return new_stack_trace


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.INFO)
    CLI(MetricsCollector, as_positional=False)
