import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import os
from collections import defaultdict
from pyparsing import line
import seutil as su
from jsonargparse import CLI
from seutil.maven import MavenModule, MavenProject
from seutil.project import Project
from tqdm import tqdm

from etestgen.data.data import DataNE2E, TData, DataMUT2E
from etestgen.data.tool import Tool
from etestgen.data.utils import save_dataset, load_dataset
from etestgen.eval.compute_throws_coverage import (
    TestMethod,
    TracedMethod,
)
from etestgen.macros import Macros
from etestgen.eval.analyze_throws_coverage import (
    extract_stmts_from_ast,
    CodeMappingException,
)
from etestgen.eval.compute_etest_coverage import Coverage
import sys

sys.setrecursionlimit(10000)

logger = su.log.get_logger(__name__)


def remove_duplicate_stack_trace(stack_trace: List):
    exist_cm = set()

    new_stack_trace = []
    for st in stack_trace[:-1]:
        if str(st) not in exist_cm:
            exist_cm.add(str(st))
            new_stack_trace.append(st)
    new_stack_trace.append(stack_trace[-1])
    return new_stack_trace


class CoverageAnalyzer:
    """Scripts for analyzing the coverage results."""

    maven_src_path = "src/main/java"

    def __init__(
        self,
        coverage_dir: su.arg.RPath = Macros.work_dir / "coverage-new",
        repos_file: su.arg.RPath = Macros.work_dir
        / "repos"
        / "filtered"
        / "repos.json",
    ):
        self.coverage_dir = coverage_dir
        self.projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        self.indexed_projects: Dict[str, Project] = {
            p.full_name: p for p in self.projects
        }
        self.downloads_dir = Macros.downloads_dir
        self.raw_data_dir = Macros.work_dir / "raw-data"

    def extract_mut2e_dataset(
        self,
        target_project_names: List[str] = None,
    ):
        Tool.ensure_tool_versions()
        Tool.require_compiled()

        records_error = []
        all_projects = list(self.coverage_dir.iterdir())
        data_id = 0
        for project_dir in tqdm(all_projects, desc="Analyzing"):
            pname = project_dir.name

            if pname not in self.indexed_projects:
                logger.info(f"Skipping {pname} not in repos list")
                continue
            if target_project_names and pname not in target_project_names:
                continue

            # iterate each module in the project
            for module_dir in project_dir.iterdir():
                if not module_dir.is_dir():
                    continue
                module_i = int(os.path.basename(module_dir))
                maven_module = su.io.load(module_dir / "module.yaml", clz=MavenModule)
                module = maven_module.rel_path
                try:
                    project = self.indexed_projects[pname]
                    module_dataset = self.extract_module_mut2e_dataset(
                        project, module, module_i, data_id
                    )
                    logger.info(
                        f"Collected {len(module_dataset)} data points from {pname}/{module}"
                    )
                    if len(module_dataset) > 0:
                        save_dataset(
                            Macros.data_dir / "mut2e-new",
                            module_dataset,
                            append=True,
                        )
                        data_id += len(module_dataset)
                except KeyboardInterrupt:
                    raise
                except:
                    logger.warning(
                        f"Error while analyzing {pname}/{module}: {traceback.format_exc()}"
                    )
                    records_error.append(
                        {
                            "project": pname,
                            "module": module,
                            "error": traceback.format_exc(),
                        }
                    )
                    continue
        #

    def extract_conditions_for_eval_data(self, dataset: str):
        """Extract conditions from real data."""
        dataset_list = load_dataset(Macros.data_dir / dataset, clz=DataNE2E)

        Tool.ensure_tool_versions()
        Tool.require_compiled()

        for data in tqdm(
            dataset_list,
            desc="Extracting conditions from stack trace",
            total=len(dataset_list),
        ):
            if len(data.e_stack_trace) == 0:
                continue
            e_stack_traces = data.e_stack_trace
            stack_trace_list = []
            for e_stack_trace in e_stack_traces:
                if len(e_stack_trace) > 70:
                    data = remove_duplicate_stack_trace(data)
                    e_stack_trace = data.e_stack_trace
                method_list, line_number_list = [], []
                for stack_trace in [e_stack_trace]:
                    method_string = stack_trace[0]["method_node"]
                    if method_string.startswith("default "):
                        method_string = method_string.replace("default ", "")
                    method_list.append(method_string)
                    line_number_list.append(stack_trace[1] + 2)
                stack_trace_list.append(
                    {
                        "methodStrings": method_list,
                        "lineNumbers": line_number_list,
                    }
                )
            stack_trace_file = Macros.data_dir / "mut2e" / "stacktrace-list.json"
            su.io.dump(stack_trace_file, stack_trace_list)
            res = su.bash.run(
                f"java -cp {Tool.core_jar} org.etestgen.core.ConditionCollector {stack_trace_file} {Macros.data_dir}/{dataset}/condition-{data.id}.jsonl",
                0,
            )
        logger.warning(res.stdout)
        su.io.rm(stack_trace_file)

        # aggregate
        for data in tqdm(dataset_list, total=len(dataset_list)):
            if len(data.e_stack_trace) == 0:
                data.condition = []
                continue
            data.condition = []
            condition_file = Macros.data_dir / dataset / f"condition-{data.id}.jsonl"
            data.condition.extend(su.io.load(condition_file))

        save_dataset(Macros.data_dir / f"{dataset}-backup", dataset_list)

    def extract_conditions(self, dataset: str):
        dataset_list = load_dataset(Macros.data_dir / dataset, clz=DataNE2E)

        Tool.ensure_tool_versions()
        Tool.require_compiled()
        stack_trace_list = []
        i = 0
        for data in tqdm(
            dataset_list, desc="Extracting e stack trace", total=len(dataset_list)
        ):
            i += 1
            e_stack_trace = data.e_stack_trace
            # if len(e_stack_trace) > 70:
            #     e_stack_trace = remove_duplicate_stack_trace(e_stack_trace)
            method_list, line_number_list = [], []
            for stack_trace in e_stack_trace:
                method_string = stack_trace[0]["method_node"]
                if method_string.startswith("default "):
                    method_string = method_string.replace("default ", "")
                method_list.append(method_string)
                line_number_list.append(stack_trace[1] + 2)
            stack_trace_list.append(
                {
                    "methodStrings": method_list,
                    "lineNumbers": line_number_list,
                }
            )
        logger.info(f"Size of stack trace list: {len(stack_trace_list)}")
        stack_trace_file = (
            Macros.data_dir / dataset / "stacktrace-list.json"
        )
        su.io.dump(stack_trace_file, stack_trace_list)
        res = su.bash.run(
            f"java -cp {Tool.core_jar} org.etestgen.core.ConditionCollector {stack_trace_file} {Macros.data_dir}/{dataset}/condition.jsonl",
            0,
        )
        logger.warning(res.stdout)

    def test_stack_trace(self):
        test_st_data = su.io.load("../collector/stacktrace-test/test-st-config.json")
        input_file_name = "../collector/stacktrace-test/test-st-config.json"
        file_name = "./conditions.txt"
        list_argument = []
        test_st_data = test_st_data
        # for mt, linum in zip(
        #     test_st_data["methodStrings"][::-1], test_st_data["lineNumbers"][::-1]
        # ):
        #     list_argument.append(f"{mt}@@{linum}")
        # argument = "**".join(list_argument)
        Tool.ensure_tool_versions()
        Tool.require_compiled()
        ps = su.bash.run(
            f"java -cp {Tool.core_jar} org.etestgen.core.ConditionCollector {input_file_name} ./conditions.txt",
            0,
        )
        print(ps.stdout)

    def extract_module_mut2e_dataset(
        self, project: Project, module_name: str, module_i: int, data_id: int
    ):
        module_dataset = []
        module_coverage_data_dir = self.coverage_dir / project.full_name / str(module_i)

        method_records = collect_module_source_code(
            project, self.downloads_dir, module_name
        )
        su.io.dump(module_coverage_data_dir / "method_records.jsonl", method_records)
        method2code = defaultdict(list)
        for record in method_records:
            ms_key = "#".join(record["method"].split("#")[:2])
            method2code[ms_key].append(record)
        #
        try:
            manual_tests = su.io.load(
                module_coverage_data_dir / "manual.tests.jsonl", clz=TestMethod
            )
            cov_data = su.io.load(
                module_coverage_data_dir / "manual.etests-coverage.jsonl", clz=Coverage
            )
        except FileNotFoundError:
            logger.warning(f"Cannot find coverage data for {project.full_name}")
            return []
        for cov_d in cov_data:
            data = DataMUT2E()
            # project info
            data.project = project.full_name
            data.module = module_name
            data.module_i = module_i

            # stact trace is a list of method
            stack_trace = extract_etest_stack_trace(cov_d, method2code)
            # remove duplicated methods in stack trace
            stack_trace = clean_stack_trace(stack_trace)
            if len(stack_trace) > 0:
                mut_dict = stack_trace[0][0]
            else:
                continue
            # mut
            data.mut = mut_dict["method_node"]
            data.mut_key = mut_dict["method"]
            mut_constructor_key = mut_dict["method"].split("#")[0] + "#<init>#"
            data.constructors = [
                method2code[mt_key]
                for mt_key in method2code
                if mut_constructor_key in mt_key
            ]
            # stack trace
            data.e_stack_trace = [(st[0], st[1]) for st in stack_trace]
            # etest
            etest_id = cov_d.test_i
            test_e = manual_tests[etest_id]
            # post processing test_e
            test_e = self.add_expected_tag(test_e, test_e.exception)
            data.test_method = test_e
            data.test_e = test_e.raw_code
            data.etest_pattern = test_e.pattern
            data.test_context = test_e.ccontext
            data.etype = test_e.exception
            data.test_e_key = (
                test_e.cname.replace(".", "/") + "#" + test_e.mname + "#()V"
            )
            data.id = data_id
            module_dataset.append(data)
            data_id += 1
        return module_dataset

    def collect_sign_stmts(self, data: TData, test_m: TestMethod) -> dict:
        # collect sign ast
        data.test_sign = test_m.ast.get_sign()
        try:
            data.test_stmts = extract_stmts_from_ast(test_m.ast)
        except CodeMappingException:
            data.test_stmts = []
        return data

    def add_expected_tag(self, test_e: str, etype: str):
        """Add expected tag to the test_e."""

        # process etype
        if "." in etype:
            etype = etype.split(".")[-1]
            if "$" in etype:
                etype = etype.split("$")[-1]
        if "@Test(expected" not in test_e.raw_code:
            test_e.raw_code = test_e.raw_code.replace(
                "@Test", f"@Test(expected = {etype}.class)"
            )
        return test_e


def clean_stack_trace(stack_trace: List):
    """
    Remove duplicated methods in stack trace.

    The method will be removed from the stack trace if:
    1. the called line number is negative
    2. there is the same method right after it

    Args:
        stack_trace (List[Tuple[dict, int]]): The input stack trace.

    Returns:
        List[Tuple[dict, int]]: The cleaned stack trace.
    """
    if len(stack_trace) <= 1:
        return stack_trace
    negative_ln = False
    for st in stack_trace:
        if int(st[1]) < 0:
            negative_ln = True
            break
    if not negative_ln:
        return stack_trace
    new_stack_trace = []
    for i, st in enumerate(stack_trace[:-1]):
        called_line_number = int(st[1])
        method = st[0]
        if called_line_number < 0 and stack_trace[i + 1][0] == method:
            continue
        new_stack_trace.append(st)
    # add the last one
    new_stack_trace.append(stack_trace[-1])
    return new_stack_trace


def extract_etest_stack_trace(cov_d: Coverage, method_record: dict):
    """
    Extract the stack trace and the method under test.
    MUT: the method at the end of stack trace
    """
    called_methods: List[TracedMethod] = cov_d.methods
    if len(called_methods) == 0:
        return []
    method_sequence = []
    found = False
    for ms in called_methods[::-1]:
        ms_key = ms.class_name + "#" + ms.method_name
        if ms_key in method_record and len(method_record[ms_key]) == 1:
            linum = int(ms.called_line_number) - int(
                method_record[ms_key][0]["startLine"]
            )
            method_sequence.append((method_record[ms_key][0], linum))
        elif ms_key in method_record and len(method_record[ms_key]) > 1:
            # find the one that matches the line number
            for possible_method in method_record[ms_key]:
                if (
                    int(possible_method["startLine"])
                    <= int(ms.called_line_number)
                    <= int(possible_method["endLine"])
                ):
                    found = True
                    linum = int(ms.called_line_number) - int(
                        possible_method["startLine"]
                    )
                    method_sequence.append((possible_method, linum))
                    break
            if not found:
                logger.warning(
                    f"Cannot find the method {ms_key} in source code. Please double check."
                )
    #
    return method_sequence


def collect_module_source_code(
    project: Project, downloads_dir: Path, module_path: str
) -> List[Any]:
    """
    Collect the source code within the module.
    """
    project.clone(downloads_dir)
    project.checkout(project.data["sha"], forced=True)
    maven_project = MavenProject.from_project(project)
    maven_module = [m for m in maven_project.modules if m.rel_path == module_path][0]

    # collect all source code in the project
    temp_dir = su.io.mktmp_dir("etestgen")
    su.io.dump(
        temp_dir / "config.json",
        {
            "mainSrcRoot": maven_module.main_srcpath,
            "classpath": maven_module.dependency_classpath,
            "outPath": str(temp_dir / "out.json"),
            "testSrcRoot": maven_module.test_srcpath,
            "debugPath": str(temp_dir / "debug.txt"),
        },
    )
    su.bash.run(
        f"java -cp {Tool.core_jar} org.etestgen.core.SrcMainMethodCollector {temp_dir}/config.json",
        0,
    )
    records_methods = su.io.load(temp_dir / "out.json")

    return records_methods


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.INFO)
    CLI(CoverageAnalyzer, as_positional=False)
