import traceback
from pathlib import Path
from typing import List, Optional, Set, Union, Any
import os
import seutil as su
import subprocess
from jsonargparse import CLI
from seutil.maven import MavenModule, MavenProject
from seutil.project import Project
from tqdm import tqdm
from collections import defaultdict
import xmltodict

from etestgen.data.tool import Tool
from etestgen.macros import Macros
from etestgen.eval.extract_etest_data_from_coverage import collect_module_source_code
from etestgen.eval.compute_throws_coverage import (
    scan_manual_tests,
    TestMethod,
    Coverage,
    process_logged_stacktraces,
)

logger = su.log.get_logger(__name__)


class TestCoverageCollector:
    DOWNLOADS_DIR = Macros.downloads_dir
    test_placeholder = "/*TEST PLACEHOLDER*/"
    instrument_log_file = "methods-logs.txt"

    def __init__(
        self,
        test_timeout: int = 15,
        project_time_limit: int = 1800,
        randoop_time_limit: int = 120,
        randoop_num_limit: int = 100000,
    ):
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.raw_data_dir = Macros.work_dir / "raw-data"

    def compute_netest_coverage(
        self,
        out_dir: Path = Macros.work_dir / "coverage-new",
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
                    self.compute_netest_coverage_for_project(p, out_dir / p.full_name)
                    success += 1
            except KeyboardInterrupt:
                raise
            except:
                logger.warning(f"Failed to process {p.full_name}")
                fail += 1
                su.io.dump(out_dir / p.full_name / "error.txt", traceback.format_exc())
            finally:
                # clean tmp dir
                su.bash.run("rm -rf /tmp/etestgen-*")
                pbar.update(1)
        pbar.set_description(f"Finished (+{success} -{fail} s{skip})")
        pbar.close()

    def compute_netest_coverage_for_project(self, project: Project, out_dir: Path):
        """
        Collect the test coverage for **NE** tests.
        """

        maven_proj = self.prepare_maven_project(project)
        su.io.dump(out_dir / "maven.yaml", maven_proj)

        # treat each module separately
        skipped = []
        for module_i, maven_module in enumerate(maven_proj.modules):
            # prepare work dir and out dir
            module_out_dir = out_dir / f"{module_i}"
            # if (module_out_dir / "netest-coverage.jsonl").exists():
            #     continue
            su.io.dump(module_out_dir / "module.yaml", maven_module)
            work_dir = su.io.mktmp_dir("etestgen")
            #  to accelerate the process, skip module with no etests
            # if not check_if_module_has_etest(module_out_dir):
            #     su.io.rmdir(work_dir)
            #     continue
            skipped += prepare_maven_module(project, maven_module, maven_proj)
            if skipped and skipped[-1][0] == maven_module.rel_path:
                su.io.rmdir(work_dir)
                continue
            # parse the source code
            try:
                method_records = su.io.load(module_out_dir / "method_records.jsonl")
            except FileNotFoundError:
                method_records = collect_module_source_code(
                    project,
                    Macros.work_dir / "downloads",
                    maven_module.rel_path,
                )
                su.io.dump(module_out_dir / "method_records.jsonl", method_records)

            method2code = defaultdict(list)
            for record in method_records:
                method2code[record["method"].replace(".", "/")].append(record)
            # collect manual tests
            if not (module_out_dir / "manual.tests.jsonl").exists():
                try:
                    scan_manual_tests(maven_module, work_dir, module_out_dir)
                except KeyboardInterrupt:
                    raise
                except:
                    logger.warning(
                        f"collecting manual tests failed for {project.full_name} {maven_module.coordinate}: {traceback.format_exc()}"
                    )
                    raise RuntimeError(
                        f"collecting manual tests failed for {project.full_name} {maven_module.coordinate}: {traceback.format_exc()}"
                    )
            # scan and instrument source
            self.scan_instrument_public_methods(maven_module, work_dir, module_out_dir)
            # run tests to collect coverage
            covered_methods = self.run_netests_compute_coverage(
                maven_module, work_dir, module_out_dir, method2code
            )
            su.io.rmdir(work_dir)
            su.io.dump(module_out_dir / "netest-coverage.jsonl", covered_methods)
        if len(skipped) > 0:
            su.io.dump(out_dir / "skipped.jsonl", skipped)

    #######
    ## Helper functions
    #######

    def run_netests_compute_coverage(
        self,
        maven_module: MavenModule,
        work_dir: Path,
        out_dir: Path,
        method2code: dict,
    ):
        """
        Run normal (non exceptional-behavior) tests and compute the coverage.
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
        for test_i, test in enumerate(tests):
            if test.pattern is not None:
                # skip etests
                continue

            # delete old log
            su.io.rm(method_log)

            # run test
            self.run_isolated_tests(test, work_dir, out_dir, classpath, test_i)
            if method_log.exists():
                # etest for unchecked exception
                called_methods: List[str] = su.io.load(method_log, su.io.Fmt.txtList)
                stack_traces = process_logged_stacktraces(called_methods)
                directly_called_methods = self.extract_directlly_called_methods(
                    stack_traces,
                    method2code,
                )
                coverage = Coverage(test_i, directly_called_methods)
                tests_traces.append(coverage)

        return tests_traces

    def extract_directlly_called_methods(
        self,
        tracedMethods: List[Any],
        src_code_methods: dict = None,
    ) -> List[str]:
        """
        Extract the test directly called methods by ignoring the synthetic methods e.g. <clinit> methods.
        """
        # extract the method under test
        min_depth = float("-inf")

        synthetic_methods = []
        directly_called_methods = []
        if len(tracedMethods) > 0:
            for traced_method in tracedMethods:
                if src_code_methods and (
                    traced_method.logged_method not in src_code_methods
                ):
                    synthetic_methods.append(traced_method.traced_methods[0])
            synthetic_methods = set(synthetic_methods)
            for traced_method in tracedMethods:
                if src_code_methods and (
                    traced_method.logged_method not in src_code_methods
                ):
                    continue
                new_stack_trace = [
                    ste
                    for ste in traced_method.traced_methods
                    if ste not in synthetic_methods and "adhoc_" not in ste.class_name
                ]
                stack_depth = len(new_stack_trace)
                if stack_depth == 0:
                    continue
                if stack_depth < min_depth or min_depth == float("-inf"):
                    min_depth = stack_depth
                    directly_called_methods = [traced_method.logged_method]
                elif stack_depth == min_depth:
                    directly_called_methods.append(traced_method.logged_method)

        return list(set(directly_called_methods))

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

    def scan_instrument_public_methods(
        self, maven_module: MavenModule, work_dir: Path, out_dir: Path
    ):
        """
        Scan and instrument all classes: public and non-abstract methods
        """

        # scan and instrument all classes
        main_config = {
            "classroot": str(maven_module.main_classpath),
            "outPath": str(work_dir / "main-out.json"),
            "tcMethodsLogPath": str(work_dir / self.instrument_log_file),
            "debugPath": str(out_dir / "debug.txt"),
            "modify": False,
            "scanThrow": False,
        }
        main_config_path = work_dir / "main-config.json"
        su.io.dump(main_config_path, main_config)
        su.bash.run(
            f"java -cp {Tool.core_jar} org.etestgen.core.ClassMainInstrumentor {main_config_path}",
            0,
        )

    def prepare_maven_project(self, project: Project) -> MavenProject:
        """
        Prepare the maven project for running experiments
        """
        # clone, checkout, clean
        project.clone(self.DOWNLOADS_DIR)
        project.checkout(project.data["sha"], forced=True)
        with su.io.cd(project.dir):
            su.bash.run("git clean -ffdx")

        # prepare the Maven stuff
        maven_proj = MavenProject.from_project(project)
        maven_proj.compile()
        maven_proj.backup_pom()
        maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")
        return maven_proj


def prepare_maven_module(
    project: Project, maven_module: MavenModule, maven_proj: MavenProject
):
    """
    Prepare the maven module for running experiments
    """
    skipped = []
    if maven_module.packaging == "pom":
        # skip parent/aggregator modules
        skipped.append((maven_module.rel_path, maven_module.coordinate, "package==pom"))
        return skipped
    if (
        not Path(maven_module.main_srcpath).exists()
        or not Path(maven_module.test_srcpath).exists()
    ):
        # skip non-standard modules
        skipped.append((maven_module.rel_path, maven_module.coordinate, "missing src"))
        return skipped

    # checkout and recompile the whole project
    maven_proj.restore_pom()
    project.checkout(project.data["sha"], forced=True)
    with su.io.cd(project.dir):
        su.bash.run("git clean -ffdx")
    maven_proj.backup_pom()
    maven_proj.hack_pom_delete_plugin("")  # enable maven debug mode
    maven_proj.compile()
    maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")
    return []


def check_if_module_has_etest(module_dir: Path):
    try:
        manual_tests = su.io.load(module_dir / "manual.tests.jsonl", clz=TestMethod)
        etests = [t for t in manual_tests if t.pattern is not None]
        return len(etests) > 0
    except FileNotFoundError:
        return True


def build_source_code_dict(maven_module: MavenModule):
    # build dict to map method id to source code

    temp_dir = su.io.mktmp_dir("etestgen")
    su.io.dump(
        temp_dir / "config.json",
        {
            "mainSrcRoot": maven_module.main_srcpath,
            "classpath": maven_module.dependency_classpath,
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
    method2code = {}
    for record in records_methods:
        method2code[record["method"]] = record["method_node"]
    su.io.rmdir(temp_dir)
    return method2code


def hack_pom_add_debug_info(module: MavenModule):
    """
    Hack the pom.xml to add the debug info while compilation
    """

    pom = xmltodict.parse(
        su.io.load(module.dir / module.rel_path / "pom.xml", fmt=su.io.Fmt.txt)
    )

    plugins = (
        pom.get("project", {}).get("build", {}).get("plugins", {}).get("plugin", [])
    )
    for plugin in plugins:
        if plugin["artifactId"] == "maven-compiler-plugin":
            plugin["configuration"]["debug"] = "true"
            plugin["configuration"]["debuglevel"] = "lines"
    su.io.dump(
        module.project.dir / module.rel_path / "pom.xml",
        xmltodict.unparse(pom),
        fmt=su.io.Fmt.txt,
    )


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.WARNING)
    CLI(TestCoverageCollector, as_positional=False)
