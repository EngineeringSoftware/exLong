import traceback
from pathlib import Path
from typing import List, Optional, Set, Union, Any, Tuple, Iterable
import os
import copy
from torch import ne
from traitlets import default
from etestgen.data.utils import load_dataset, save_dataset
import seutil as su
import subprocess
from statistics import mean
from jsonargparse import CLI
from seutil.maven import MavenModule, MavenProject
from seutil.project import Project
from tqdm import tqdm
from collections import defaultdict

from etestgen.data.tool import Tool
from etestgen.data.data import DataNE2E
from etestgen.data.utils import save_dataset
from etestgen.macros import Macros
from etestgen.eval.compute_throws_coverage import (
    TestMethod,
    Coverage,
    process_logged_stacktraces,
)
from etestgen.eval.compute_test_coverage import (
    check_if_module_has_etest,
    prepare_maven_module,
)
from etestgen.data.structures import AST, ClassStructure, Consts, MethodStructure, Scope
from etestgen.eval.compute_throws_coverage import (
    Coverage,
    TestMethod,
    StackTrace,
    TracedMethod,
)
from etestgen.eval.extract_etest_data_from_coverage import clean_stack_trace

logger = su.log.get_logger(__name__)


class TestStackTraceCollector:
    DOWNLOADS_DIR = Macros.downloads_dir
    test_placeholder = "/*TEST PLACEHOLDER*/"
    instrument_log_file = "methods-logs.txt"
    method_ex_to_lines = defaultdict(list)

    def __init__(
        self,
        test_timeout: int = 15,
        project_time_limit: int = 3600,
        logged_methods_limit: int = 200,
        randoop_time_limit: int = 120,
        randoop_num_limit: int = 100000,
    ):
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.raw_data_dir = Macros.work_dir / "raw-data"

    def compute_netest_coverage(
        self,
        out_dir: su.arg.RPath,
        repos_file: Path = Macros.work_dir / "repos" / "filtered" / "repos.json",
        project_names: Optional[List[str]] = None,
    ):
        Tool.ensure_tool_versions()
        Tool.require_compiled()

        # load projects
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        projects.sort(key=lambda p: p.full_name)
        success, fail, skip = 0, 0, 0
        # limit to user specified projects
        if project_names is not None:
            projects = [p for p in projects if p.full_name in project_names]
            logger.info(
                f"Selected {len(projects)} projects: {[p.full_name for p in projects]}"
            )
        pbar = tqdm(total=len(projects))
        for p in projects:
            pbar.set_description(
                f"Processing {p.full_name} (+{success} -{fail} s{skip})"
            )
            try:
                su.io.mkdir(out_dir / p.full_name)
                with su.TimeUtils.time_limit(self.config["project_time_limit"]):
                    self.extract_stack_trace_for_project(p, out_dir / p.full_name)
                    success += 1
            except KeyboardInterrupt:
                raise
            except:
                logger.warning(f"Failed to process {p.full_name}")
                fail += 1
                su.io.dump(
                    out_dir / p.full_name / "throw-stack-trace-error.txt",
                    traceback.format_exc(),
                )
            finally:
                # clean tmp dir
                su.bash.run("rm -rf /tmp/etestgen-*")
                pbar.update(1)
        pbar.set_description(f"Finished (+{success} -{fail} s{skip})")
        pbar.close()

    def collect_stack_traces_from_netests(
        self,
        out_dir: su.arg.RPath,
        repos_file: Path = Macros.work_dir / "repos" / "filtered" / "repos.json",
        target_project_names: Optional[List[str]] = None,
    ):
        Tool.ensure_tool_versions()
        Tool.require_compiled()

        # load projects
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        projects.sort(key=lambda p: p.full_name)
        success, fail, skip = 0, 0, 0
        # limit to user specified projects
        if target_project_names is not None:
            projects = [p for p in projects if p.full_name in target_project_names]
            logger.info(
                f"Selected {len(projects)} projects: {[p.full_name for p in projects]}"
            )
        pbar = tqdm(total=len(projects))
        for p in projects:
            pbar.set_description(
                f"Processing {p.full_name} (+{success} -{fail} s{skip})"
            )
            try:
                with su.TimeUtils.time_limit(self.config["project_time_limit"]):
                    self.collect_project_stack_trace_with_etypes(
                        p, out_dir=out_dir / p.full_name
                    )
                    success += 1
                    su.io.rm(
                        out_dir / p.full_name / "error-netest-stack-trace.txt",
                    )
            except:
                logger.warning(f"Failed to process {p.full_name}")
                fail += 1
                su.io.dump(
                    out_dir / p.full_name / "error-netest-stack-trace.txt",
                    traceback.format_exc(),
                )
            finally:
                pbar.update(1)
        pbar.set_description(f"Finished (+{success} -{fail} s{skip})")
        pbar.close()

    def add_real_stack_traces_to_eval_data(
        self,
        out_dir: Path,
        repos_file: Path,
        data_dir: str = "ne2e-test",
    ):
        """
        Add stack traces collected from netests as eval dataset stack traces
        """
        # load project data
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        new_test_mut2e_data = []
        project_map = {p.full_name: p for p in projects}
        mut2e_data = load_dataset(Macros.data_dir / data_dir, clz=DataNE2E)
        # compile tool
        Tool.ensure_tool_versions()
        Tool.require_compiled()
        # categorize the dataset by project,module
        pm2data = defaultdict(list)
        for dt in mut2e_data:
            pm2data[(dt.project, dt.module_i)].append(dt)
        #
        with tqdm(total=len(pm2data), desc="Adding real stack traces..") as pbar:
            for pm_key in pm2data:
                p_name, module_i = pm_key
                # load method record
                method_records = su.io.load(
                    out_dir / p_name / str(module_i) / "method_records.jsonl"
                )
                method2code = {}
                for record in method_records:
                    ms_key = record["method"]
                    method2code[ms_key] = record
                #
                project = project_map[p_name]
                project.clone(downloads_dir=self.DOWNLOADS_DIR)
                maven_project = MavenProject.from_project(project)
                ne_stack_traces = su.io.load(
                    out_dir / p_name / str(module_i) / "netest-stacktrace2e.jsonl"
                )
                target_maven_module = maven_project.modules[module_i]
                for dt in pm2data[pm_key]:
                    etype = dt.etype
                    mut_key = dt.mut_key
                    possible_stack_traces = (
                        self.extract_possible_exception_stack_traces(
                            ne_stack_traces,
                            mut_key,
                            etype,
                            method2code,
                            maven_module=target_maven_module,
                        )
                    )
                    dt.e_stack_trace = copy.deepcopy(possible_stack_traces)
                    new_test_mut2e_data.append(dt)
                pbar.update(1)
        save_dataset(Macros.data_dir / f"{data_dir}-new", new_test_mut2e_data)

    def extract_possible_exception_stack_traces(
        self,
        ne_stack_traces: List[Any],
        mut_key: str,
        etype: str,
        method_record_dict: dict,
        maven_module: MavenModule,
    ) -> List:
        """
        Iterate all stack traces logged by netests, return the 'useful' ones.
        """
        possible_stack_traces = []
        if etype in method_record_dict[mut_key]["exceptions"]:
            exception_lines = extract_line_num_of_throw_statement(
                method=method_record_dict[mut_key]["method_node"],
                etype=etype,
                maven_module=maven_module,
            )
            for line_num in exception_lines:
                cleaned_stack_trace = [method_record_dict[mut_key], line_num]
                if cleaned_stack_trace not in possible_stack_traces:
                    possible_stack_traces.append(cleaned_stack_trace)
        for st_trace in ne_stack_traces:
            st_etypes = [st_e.split("@@")[0] for st_e in st_trace[1]]
            st_etypes_set = set([st_e.split(".")[-1] for st_e in st_etypes])
            if etype not in st_etypes or etype.split(".")[-1] not in st_etypes_set:
                continue
            called_methods = st_trace[0]
            for i, called_ms in enumerate(called_methods):
                if mut_key == called_ms[0]["method"]:
                    cleaned_stack_trace = called_methods[i:]
                    last_method = cleaned_stack_trace[-1][0]["method"]
                    exception_line_key = last_method + "#" + etype
                    if exception_line_key in self.method_ex_to_lines:
                        exception_lines = self.method_ex_to_lines[exception_line_key]
                    else:
                        exception_lines = extract_line_num_of_throw_statement(
                            method=cleaned_stack_trace[-1][0]["method_node"],
                            etype=etype,
                            maven_module=maven_module,
                        )
                        self.method_ex_to_lines[exception_line_key] = exception_lines
                    if exception_lines == []:
                        logger.warning("Cannot find exception line, might be buggy..")
                    for line_num in exception_lines:
                        cleaned_stack_trace[-1][1] = line_num
                        if cleaned_stack_trace not in possible_stack_traces:
                            possible_stack_traces.append(cleaned_stack_trace)
        return possible_stack_traces[0]

    def create_ne2e_test_data(self):
        ne2e_data = load_dataset(Macros.data_dir / "ne2e", clz=DataNE2E)
        test_ne2e_data = load_dataset(Macros.data_dir / "ne2e-test", clz=DataNE2E)
        test_projects = su.io.load(Macros.results_dir / "repos" / "test-projects.json")
        # split dataset by project,module
        test_ne_data = []
        for dt in ne2e_data:
            if dt.project not in test_projects:
                continue
            else:
                test_ne_data.append(dt)

        print(f"Total {len(test_ne_data)} test data points.")

        with_throw_statement = 0
        with_throw_id = []
        for dt in test_ne_data:
            last_method_lines = dt.e_stack_trace[-1][0]["method_node"].splitlines()
            line_number = dt.e_stack_trace[-1][1]

            if "throw" in last_method_lines[line_number]:
                with_throw_statement += 1
                with_throw_id.append(dt.id)
        print(f"Total {with_throw_statement} test data points with throw statement.")

        real_stack_trace_count = 0
        for dt in test_ne2e_data:
            if dt.id not in with_throw_id:
                continue
            if len(dt.e_stack_trace) > 0:
                real_stack_trace_count += 1
        print(f"Total {real_stack_trace_count} test data points with real stack trace.")
        #
        # save_dataset(Macros.data_dir / "ne2e-test", test_ne_data)
        # # copy the conditions and stack trace to this directory
        # su.bash.run(
        #     f"cp {Macros.data_dir}/mut2e-test/e_stack_trace.jsonl {Macros.data_dir}/ne2e-test/",
        #     0,
        # )
        # su.bash.run(
        #     f"cp {Macros.data_dir}/mut2e-test/condition.jsonl {Macros.data_dir}/ne2e-test/",
        #     0,
        # )

    ####
    # Helper methods
    ####

    def find_the_line_with_throw_statement(
        self, method: str, etype: str, all_etypes: List[str]
    ):
        candidate_lines = []
        exception_lines = []
        method_lines = method.splitlines()
        for i, line in enumerate(method_lines):
            if "throw " in line:
                candidate_lines.append(i)

        # further filter
        short_etype = etype.split(".")[-1].split("$")[-1]
        if len(candidate_lines) > 1:
            for i in candidate_lines:
                if short_etype in method_lines[i]:
                    exception_lines.append(i)
                if short_etype == "RuntimeException" and "rethrow" in method_lines[i]:
                    exception_lines.append(i)
                if len(exception_lines) == 0:
                    e_i = all_etypes.index(etype)
                    exception_lines.append(candidate_lines[e_i])
        else:
            exception_lines = candidate_lines

        return exception_lines

    def collect_project_stack_trace_with_etypes(
        self, project: Project, out_dir: Path
    ) -> List[DataNE2E]:
        """
        Construct coverage-stack to exception dataset for project by running existing normal tests.
        """
        # prepare maven module
        maven_proj = prepare_maven_project(project)
        # prepare collected source code
        for module_i, maven_module in enumerate(maven_proj.modules):
            # skip modules
            if maven_module.packaging == "pom":
                # skip parent/aggregator modules
                continue
            if (
                not Path(maven_module.main_srcpath).exists()
                or not Path(maven_module.test_srcpath).exists()
            ):
                continue
            module_out_dir = out_dir / f"{module_i}"
            if not check_if_module_has_etest(module_out_dir):
                continue
            try:
                covstacks = su.io.load(
                    module_out_dir / "netest-stacktrace.jsonl",
                    clz=Coverage,
                    iter_line=True,
                )
                # method_records = collect_module_source_code(
                #     project, self.DOWNLOADS_DIR, module_path=maven_module.rel_path
                # )
                # su.io.dump(module_out_dir / "method_records.jsonl", method_records)
                method_records = su.io.load(module_out_dir / "method_records.jsonl")
                method2code = defaultdict(list)
                for record in method_records:
                    ms_key = "#".join(record["method"].split("#")[:2])
                    method2code[ms_key].append(record)
                # get a list of (stack-trace:List[method,int], List[etype])
                module_covstack2e_list = self.get_module_test_stacks(
                    covstacks, method2code
                )
                su.io.dump(
                    module_out_dir / "netest-stacktrace2e.jsonl",
                    list(module_covstack2e_list),
                )
            except FileNotFoundError as e:
                logger.warning(
                    f"Failed to find the netest-stacktrace.jsonl for {project.full_name}.{module_i}"
                )

    def construct_ne2stack_module_data(
        self,
        module_covstack2e_list: List[Any],
        method2record: dict,
        manual_tests: List[TestMethod],
        module_i: int,
        project: Project,
    ):
        """
        Construct DataNE2E dataset given collecte raw data
        """
        module_data_list = []
        # Construct data point
        for test_id, mut, stack_trace in tqdm(
            module_covstack2e_list,
            total=len(module_covstack2e_list),
            desc="Constructing dataset...",
        ):
            mte = stack_trace[-1]
            method_desc = mte.cls_name + "#" + mte.namedesc
            exceptions_list = []
            if method_desc in method2record:
                exceptions_list = list(set(method2record[method_desc]["exceptions"]))
            for excep in exceptions_list:
                # construct data point
                data_info = (
                    mut,
                    stack_trace,
                    manual_tests[test_id],
                    excep,
                    module_i,
                    project.full_name,
                )
                covstack2e_dt = self.construct_datapoint(data_info)
                module_data_list.append(covstack2e_dt)
        return module_data_list

    def get_module_test_stacks(self, covstacks: List[Coverage], module_records: dict):
        module_covstack2e_list = []
        for cov_dt in tqdm(
            covstacks,
            desc="pairing stack traces with etypes...",
        ):
            covstack2dt_list = self.pair_stack_traces_with_etypes(
                cov_dt, module_records
            )
            module_covstack2e_list.extend(covstack2dt_list)
        return module_covstack2e_list

    def deduplicate_covstack2e_data(
        self, data_ne2e_list: List[DataNE2E]
    ) -> List[DataNE2E]:
        """
        De-duplicate the covstack2e data w.r.t. the mut, etype, and stack trace.
        """
        key_2_data = {}
        unique_data_list = []
        for dt in data_ne2e_list:
            e_stack_trace_str = "-".join([m.code for m in dt.call_stacks])
            key = f"{dt.mut}-{dt.etype}-{e_stack_trace_str}"
            if key not in key_2_data:
                key_2_data[key] = dt
            else:
                assert len(dt.test_ne) == 1
                if dt.test_ne[0] not in key_2_data[key].test_ne:
                    key_2_data[key].netest_methods.extend(dt.netest_methods)
                    key_2_data[key].test_ne.extend(dt.test_ne)
                    key_2_data[key].test_ne_key.extend(dt.test_ne_key)
                    key_2_data[key].netest_context.extend(dt.netest_context)
                    assert len(key_2_data[key].netest_context) == len(
                        key_2_data[key].test_ne_key
                    )
        #
        unique_data_list = list(key_2_data.values())

        return unique_data_list

    def construct_datapoint(self, data_tuple: Tuple) -> DataNE2E:
        """
        Construct the DataNE2E data point.
        """
        mut, stack_trace, test_method, exception, module_i, project_name = data_tuple
        # construct data point
        e_stack_trace = [method for method in stack_trace]
        # construct data point
        dt_point = DataNE2E(
            project=project_name,
            mut_key=mut.key.replace(".", "/"),
            module_i=module_i,
            mut=mut.code,
            etype=exception,
            call_stacks=e_stack_trace,
            netest_methods=[test_method],
            test_ne_key=[
                test_method.cname.replace(".", "/") + "#" + test_method.mname + "#()V"
            ],
            test_ne=[test_method.raw_code],
            netest_context=[test_method.ccontext],
        )
        return dt_point

    def pair_stack_traces_with_etypes(
        self,
        cov_dt: Coverage,
        method_records: dict,
    ) -> List[Tuple]:
        """
        Pair the stack traces with possible etypes
        """
        covstack2e_list = []
        deduplicate_dict = {}
        for logged_dt in cov_dt.methods:
            stack_trace_with_methods, etypes = extract_methods_traces_from_logs(
                logged_dt, method_records
            )
            if etypes:
                if str((stack_trace_with_methods, etypes)) not in deduplicate_dict:
                    deduplicate_dict[str((stack_trace_with_methods, etypes))] = 1
                    covstack2e_list.append((stack_trace_with_methods, etypes))

        return covstack2e_list

    def extract_stack_trace_for_project(self, project: Project, out_dir: Path):
        """
        Collect the test coverage for **NE** tests.
        """

        maven_proj = prepare_maven_project(project)
        su.io.dump(out_dir / "maven.yaml", maven_proj)

        # treat each module separately
        skipped = []
        for module_i, maven_module in enumerate(maven_proj.modules):
            # prepare work dir and out dir
            module_out_dir = out_dir / f"{module_i}"
            su.io.dump(module_out_dir / "module.yaml", maven_module)
            work_dir = su.io.mktmp_dir("etestgen")
            #  to accelerate the process, skip module with no etests
            if not check_if_module_has_etest(module_out_dir):
                continue
            skipped += prepare_maven_module(project, maven_module, maven_proj)
            if skipped and skipped[-1][0] == maven_module.rel_path:
                su.io.rmdir(work_dir)
                continue
            # scan and instrument throw statement
            self.scan_instrument_throw_statements(
                maven_module, work_dir, module_out_dir
            )
            # run netests to collect all logged stack traces
            covered_methods = self.run_netests_find_stacktrace(
                maven_module, work_dir, module_out_dir
            )
            su.io.rmdir(work_dir)
            su.io.dump(module_out_dir / "netest-stacktrace.jsonl", covered_methods)
        if len(skipped) > 0:
            su.io.dump(out_dir / "skipped.jsonl", skipped)

    def scan_instrument_throw_statements(
        self, maven_module: MavenModule, work_dir: Path, out_dir: Path
    ):
        """
        Scan and instrument all throw statements in the methods.
        """

        # scan and instrument all classes
        main_config = {
            "classroot": str(maven_module.main_classpath),
            "outPath": str(work_dir / "main-out.json"),
            "tcMethodsLogPath": str(work_dir / self.instrument_log_file),
            "debugPath": str(out_dir / "debug.txt"),
            "modify": False,
            "scanThrow": True,
        }
        main_config_path = work_dir / "main-config.json"
        su.io.dump(main_config_path, main_config)
        su.bash.run(
            f"java -cp {Tool.core_jar} org.etestgen.core.ThrowStatementInstrumentor {main_config_path}",
            0,
        )

    def run_netests_find_stacktrace(
        self,
        maven_module: MavenModule,
        work_dir: Path,
        out_dir: Path,
    ):
        """
        Run normal (non exceptional-behavior) tests and extract logged stack traces [the stack traces should result in invoke of method with throw statement].
        """

        classpath = os.pathsep.join(
            [
                maven_module.main_classpath,
                maven_module.test_classpath,
                maven_module.dependency_classpath,
            ]
        )

        tests: List[TestMethod] = su.io.load(
            out_dir / f"manual.tests.jsonl", clz=TestMethod
        )

        method_log = work_dir / self.instrument_log_file
        tests_traces = []
        for test_i, test in tqdm(enumerate(tests), total=len(tests)):
            # skip etests
            if test.pattern is not None:
                continue

            # delete old log
            su.io.rm(method_log)

            # run test
            self.run_isolated_tests(test, work_dir, out_dir, classpath, test_i)
            if method_log.exists():
                called_methods: List[str] = su.io.load(method_log, su.io.Fmt.txtList)
                stack_traces = process_logged_stacktraces(called_methods)
                coverage = Coverage(test_i, stack_traces)
                tests_traces.append(coverage)

        return tests_traces

    def run_isolated_tests(
        self,
        test: TestMethod,
        work_dir: Path,
        out_dir: Path,
        classpath: Any,
        test_log_name: Any,
    ):
        """
        Run isolated tests.
        """

        # extract the test to an ad-hoc test class
        run_path = work_dir / "run"
        su.io.mkdir(run_path, fresh=True)
        ccontext = test.ccontext
        if test.ccontext is None:
            package = None
            test_name = "adhoc_Test"
            test_path = run_path / f"{test_name}.java"
            ccontext = "public class adhoc_Test { " + self.test_placeholder + " }"
        else:
            package = ".".join(test.cname.split(".")[0:-1])
            csname = "adhoc_" + test.cname.split(".")[-1]
            test_name = package + "." + csname
            test_path = run_path / package.replace(".", "/") / f"{csname}.java"
            ccontext = test.ccontext
            ccontext = ccontext.replace(
                "import static org.mockito.Mockito.when;",
                "import static org.mockito.Mockito.when;\nimport org.mockito.Mockito;\n",
            )
            ccontext = ccontext.replace("when(", "Mockito.lenient().when(")

        test_file_content = f"public void adhoc_test() throws Exception {test.code}\n"
        test_file_content = f"@org.junit.Test\n" + test_file_content
        test_file_content = ccontext.replace(self.test_placeholder, test_file_content)
        su.io.dump(test_path, test_file_content, su.io.Fmt.txt)

        with su.io.cd(run_path):
            # compile the test
            rr = su.bash.run(f"javac -cp {classpath} {test_path}")

            if rr.returncode != 0:
                su.io.dump(
                    out_dir / f"errors" / f"{test_log_name}.java",
                    test_file_content,
                    su.io.Fmt.txt,
                )
                su.io.dump(
                    out_dir / f"errors" / f"{test_log_name}.log",
                    "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                    su.io.Fmt.txt,
                )

            # run the test
            try:
                rr = su.bash.run(
                    f"java -cp .:{Tool.rt_jar}:{classpath} -ea org.junit.runner.JUnitCore {test_name}",
                    timeout=self.config["test_timeout"],
                )
            except subprocess.TimeoutExpired:
                su.io.dump(
                    out_dir / f"errors" / f"{test_log_name}.java",
                    test_file_content,
                    su.io.Fmt.txt,
                )
                su.io.dump(
                    out_dir / f"errors" / f"{test_log_name}.log",
                    "TIMEOUT",
                    su.io.Fmt.txt,
                )

            if rr.returncode != 0:
                su.io.dump(
                    out_dir / f"errors" / f"{test_log_name}.java",
                    test_file_content,
                    su.io.Fmt.txt,
                )
                su.io.dump(
                    out_dir / f"errors" / f"{test_log_name}.log",
                    "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                    su.io.Fmt.txt,
                )
        su.io.rmdir(run_path)


def extract_line_num_of_throw_statement(
    method: str, etype: str, maven_module: MavenModule
) -> List[int]:
    """
    Use JavaParser to extract the list of line numbers that contains expected throw statement.
    """
    # start collect data
    if "throw" not in method:
        return []
    if method.startswith("default "):
        method = method.replace("default ", "")
    temp_dir = su.io.mktmp_dir("etestgen")
    su.io.dump(
        temp_dir / "config.json",
        {
            "methodString": method,
            "mainSrcRoot": maven_module.main_srcpath,
            "classpath": maven_module.dependency_classpath,
            "outPath": str(temp_dir / "out.json"),
            "testSrcRoot": maven_module.test_srcpath,
            "debugPath": str(temp_dir / "debug.txt"),
        },
    )
    try:
        su.bash.run(
            f"java -cp {Tool.core_jar} org.etestgen.core.SrcMainThrowCollector {temp_dir}/config.json",
            0,
        )
    except Exception:
        logger.warning(f"Failed to extract throw statement line number for {method}")
        raise RuntimeError
    recorded_exception_lines = su.io.load(temp_dir / "out.json")[0]
    su.io.rmdir(temp_dir)
    if etype in recorded_exception_lines:
        return recorded_exception_lines[etype]
    else:
        lines = []
        for k, v in recorded_exception_lines.items():
            if etype.split(".")[-1] in k:
                lines.extend(v)

        return lines


def extract_methods_traces_from_logs(
    logged_trace: StackTrace,
    method_dict: dict,
) -> Tuple[List[Tuple[Any]], List]:
    """
    Extract a list (method, linum) from the logged traces after running the tests.

    Return the list of stack trace method with relative called line number.
    """

    stack_trace = logged_trace.traced_methods[::-1]

    called_method_traces = []
    stack_trace_2_method = {}
    etypes = None
    for i, cm in enumerate(stack_trace):
        frame_key = f"{cm.class_name}#{cm.method_name}#{cm.called_line_number}"
        if frame_key in stack_trace_2_method:
            called_method_traces.append(stack_trace_2_method[frame_key])
            if i == len(stack_trace) - 1:
                etypes = stack_trace_2_method[frame_key][0]["exceptions"]
            continue
        key = f"{cm.class_name}#{cm.method_name}"
        candidate_methods = method_dict[key]
        if len(candidate_methods) == 1:
            if cm.called_line_number > 0:
                relative_called_line = int(cm.called_line_number) - int(
                    candidate_methods[0]["startLine"]
                )
            else:
                relative_called_line = -1
            called_method_traces.append((candidate_methods[0], relative_called_line))
            stack_trace_2_method[frame_key] = (
                candidate_methods[0],
                relative_called_line,
            )
            if i == len(stack_trace) - 1:
                etypes = candidate_methods[0]["exceptions"]
        elif len(candidate_methods) == 0:
            break
        else:
            # if multiple candidate, check the line number
            if cm.called_line_number < 0:
                if i == len(stack_trace) - 1:
                    # this is the one logged so the line number is -1
                    for possible_method in candidate_methods:
                        if (
                            possible_method["method"].replace(".", "/")
                            == logged_trace.logged_method
                        ):
                            called_method_traces.append((possible_method, -1))
                            stack_trace_2_method[frame_key] = (
                                candidate_methods[0],
                                -1,
                            )
                            if i == len(stack_trace) - 1:
                                etypes = possible_method["exceptions"]
                            break
                else:
                    logger.warning(
                        f"called line number is negative. Please double check."
                    )

            else:
                # search by line number
                found = False
                for possible_method in candidate_methods:
                    if (
                        int(possible_method["startLine"])
                        <= int(cm.called_line_number)
                        <= int(possible_method["endLine"])
                    ):
                        found = True
                        linum = int(cm.called_line_number) - int(
                            possible_method["startLine"]
                        )
                        called_method_traces.append((possible_method, linum))
                        stack_trace_2_method[frame_key] = (
                            candidate_methods[0],
                            linum,
                        )
                        if i == len(stack_trace) - 1:
                            etypes = possible_method["exceptions"]
                        break
                if not found:
                    logger.warning(
                        f"Cannot find the method {cm} in source code even given the line number. Probably because of override."
                    )
    return (called_method_traces, etypes)


def prepare_maven_project(project: Project) -> MavenProject:
    """
    Prepare the maven project for running experiments
    """
    # clone, checkout, clean
    project.clone(Macros.downloads_dir)
    project.checkout(project.data["sha"], forced=True)
    with su.io.cd(project.dir):
        su.bash.run("git clean -ffdx")

    # prepare the Maven stuff
    maven_proj = MavenProject.from_project(project)
    maven_proj.backup_pom()
    maven_proj.hack_pom_delete_plugin("")
    maven_proj.compile()
    maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")
    return maven_proj


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.INFO)
    CLI(TestStackTraceCollector, as_positional=False)
