import dataclasses
import os
import re
import subprocess
import traceback
from pathlib import Path
from typing import List, Optional, Set, Union
from copy import deepcopy

import seutil as su
from jsonargparse import CLI
from seutil.maven import MavenModule, MavenProject
from seutil.project import Project
from tqdm import tqdm

from etestgen.data.tool import Tool
from etestgen.data.cg import CallGraph, EdgeType
from etestgen.data.structures import (
    ClassStructure,
    Insn,
    MethodStructure,
)
from etestgen.data.structures import AST
from etestgen.macros import Macros

logger = su.log.get_logger(__name__)


@dataclasses.dataclass(unsafe_hash=True)
class ThrowsClause:
    method: str
    exception: str
    has_condition: bool = False


@dataclasses.dataclass
class ThrowsClauseMethod:
    method: str
    call_stacks: List[List[int]]
    exception: str


@dataclasses.dataclass
class TracedMethod:
    class_name: str = None
    method_name: str = None
    called_line_number: int = None

    def __hash__(self):
        return hash((self.method_name, self.class_name, self.called_line_number))


@dataclasses.dataclass
class StackTrace:
    stack_depth: int = None
    logged_method: str = None
    traced_methods: List[TracedMethod] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TestMethod:
    cname: str = None
    mname: str = None
    ast: AST = None
    raw_code: str = None
    code: str = None
    exception: Optional[str] = None
    pattern: Optional[str] = None
    ccontext: Optional[str] = None
    comment: Optional[str] = None
    commentSummary: Optional[str] = None
    context: Optional[str] = None


@dataclasses.dataclass
class Coverage:
    test_i: int
    methods: List[StackTrace] = dataclasses.field(default_factory=list)
    # this should contain the methods that throw exceptions
    mte: List[StackTrace] = dataclasses.field(default_factory=list)
    # the existing code in the test class
    test_context: str = None


def scan_manual_tests(maven_module: MavenModule, work_dir: Path, out_dir: Path):
    test_config = {
        "classpath": maven_module.exec_classpath,
        "mainSrcRoot": maven_module.main_srcpath,
        "testSrcRoot": maven_module.test_srcpath,
        "outPath": str(work_dir / "manual.out.json"),
        "debugPath": str(out_dir / "debug.txt"),
    }
    test_config_path = work_dir / "manual.config.json"
    su.io.dump(test_config_path, test_config)
    su.bash.run(
        f"java -cp {Tool.core_jar} org.etestgen.core.SrcTestScanner {test_config_path}",
        0,
    )
    tests = su.io.load(work_dir / "manual.out.json", clz=List[TestMethod])
    su.io.dump(out_dir / "manual.tests.jsonl", tests)
    return tests


class ThrowsCoverageComputer:
    DOWNLOADS_DIR = Macros.teco_downloads_dir

    def __init__(
        self,
        test_timeout: int = 15,
        project_time_limit: int = 1800,
        randoop_time_limit: int = 120,
        randoop_num_limit: int = 100000,
    ):
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.raw_data_dir = Macros.work_dir / "raw-data"

    def scan_tests(
        self,
        out_dir: su.arg.RPath,
        repos_file: su.arg.RPath = Macros.work_dir
        / "repos"
        / "filtered"
        / "mut2e-repos.json",
        project_names: Optional[List[str]] = None,
        skip_project_names: Optional[List[str]] = None,
        overwrite: bool = True,
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
        if skip_project_names is not None:
            new_projects = [
                p for p in projects if p.full_name not in skip_project_names
            ]
            skip = len(projects) - len(new_projects)
            logger.info(f"Skipped {skip} projects")
            projects = new_projects

        pbar = tqdm(total=len(projects))
        for p in projects:
            pbar.set_description(
                f"Processing {p.full_name} (+{success} -{fail} s{skip})"
            )
            try:
                su.io.mkdir(out_dir / p.full_name)
                with su.TimeUtils.time_limit(self.config["project_time_limit"]):
                    self.scan_manual_tests_from_project(p, out_dir / p.full_name)
                    success += 1
            except KeyboardInterrupt:
                raise
            except:
                logger.warning(f"Failed to process {p.full_name}")
                fail += 1
                su.io.dump(out_dir / p.full_name / "error.txt", traceback.format_exc())
            finally:
                pbar.update(1)
        pbar.set_description(f"Finished (+{success} -{fail} s{skip})")
        pbar.close()

    def scan_manual_tests_from_project(self, project: Project, out_dir: Path):
        """
        Scan and write down the manual-written tests.
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
        su.io.dump(out_dir / "maven.yaml", maven_proj)

        # treat each module separately
        skipped = []
        for module_i, maven_module in enumerate(maven_proj.modules):
            if maven_module.packaging == "pom":
                # skip parent/aggregator modules
                skipped.append((module_i, maven_module.coordinate, "package==pom"))
                continue

            if (
                not Path(maven_module.main_srcpath).exists()
                or not Path(maven_module.test_srcpath).exists()
            ):
                # skip non-standard modules
                skipped.append((module_i, maven_module.coordinate, "missing src"))
                continue

            # checkout and recompile the whole project
            maven_proj.restore_pom()
            project.checkout(project.data["sha"], forced=True)
            with su.io.cd(project.dir):
                su.bash.run("git clean -ffdx")
            maven_proj.compile()
            maven_proj.backup_pom()
            maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")

            # prepare work dir and out dir
            work_dir = su.io.mktmp_dir("etestgen")
            module_out_dir = out_dir / f"{module_i}"
            su.io.dump(module_out_dir / "module.yaml", maven_module)

            # collect manual tests
            success_kinds = []
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
            else:
                success_kinds.append("manual")
            su.io.dump(module_out_dir / "success-kinds.json", success_kinds)

    def compute_project(self, project: Project, out_dir: Path):
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
        su.io.dump(out_dir / "maven.yaml", maven_proj)

        # treat each module separately
        skipped = []
        for module_i, maven_module in enumerate(maven_proj.modules):
            if maven_module.packaging == "pom":
                # skip parent/aggregator modules
                skipped.append((module_i, maven_module.coordinate, "package==pom"))
                continue

            if (
                not Path(maven_module.main_srcpath).exists()
                or not Path(maven_module.test_srcpath).exists()
            ):
                # skip non-standard modules
                skipped.append((module_i, maven_module.coordinate, "missing src"))
                continue

            # checkout and recompile the whole project
            maven_proj.restore_pom()
            project.checkout(project.data["sha"], forced=True)
            with su.io.cd(project.dir):
                su.bash.run("git clean -ffdx")
            maven_proj.compile()
            maven_proj.backup_pom()
            maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")

            # prepare work dir and out dir
            work_dir = su.io.mktmp_dir("etestgen")
            module_out_dir = out_dir / f"{module_i}"
            su.io.dump(module_out_dir / "module.yaml", maven_module)

            # collect manual tests
            success_kinds = []
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
            else:
                success_kinds.append("manual")
            # scan and instrument source
            throws_clauses = self.scan_instrument_throws_clauses(
                maven_module, work_dir, module_out_dir
            )
            su.io.dump(module_out_dir / "success-kinds.json", success_kinds)
            potential_mnames = self.get_potential_mnames(throws_clauses)
            for kind in success_kinds:
                self.run_etests_compute_coverage(
                    maven_module, kind, work_dir, module_out_dir, potential_mnames
                )
                # self.run_netests_compute_coverage(
                #     maven_module, kind, work_dir, module_out_dir
                # )

            su.io.rmdir(work_dir)
        if len(skipped) > 0:
            su.io.dump(out_dir / "skipped.jsonl", skipped)

    def compute_netest_coverage(self, project: Project, out_dir: Path):
        """
        Collect the test coverage for **NE** tests.
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
        su.io.dump(out_dir / "maven.yaml", maven_proj)

        # treat each module separately
        skipped = []
        for module_i, maven_module in enumerate(maven_proj.modules):
            if maven_module.packaging == "pom":
                # skip parent/aggregator modules
                skipped.append((module_i, maven_module.coordinate, "package==pom"))
                continue

            if (
                not Path(maven_module.main_srcpath).exists()
                or not Path(maven_module.test_srcpath).exists()
            ):
                # skip non-standard modules
                skipped.append((module_i, maven_module.coordinate, "missing src"))
                continue

            # checkout and recompile the whole project
            maven_proj.restore_pom()
            project.checkout(project.data["sha"], forced=True)
            with su.io.cd(project.dir):
                su.bash.run("git clean -ffdx")
            maven_proj.compile()
            maven_proj.backup_pom()
            maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")

            # prepare work dir and out dir
            work_dir = su.io.mktmp_dir("etestgen")
            module_out_dir = out_dir / f"{module_i}"
            su.io.dump(module_out_dir / "module.yaml", maven_module)

            # collect manual tests
            success_kinds = []
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
            else:
                success_kinds.append("manual")
            # scan and instrument source
            throws_clauses = self.scan_instrument_throws_clauses(
                maven_module, work_dir, module_out_dir
            )
            su.io.dump(module_out_dir / "success-kinds.json", success_kinds)
            for kind in success_kinds:
                self.run_netests_compute_coverage(
                    maven_module, kind, work_dir, module_out_dir
                )

            su.io.rmdir(work_dir)
        if len(skipped) > 0:
            su.io.dump(out_dir / "skipped.jsonl", skipped)

    def scan_instrument_throws_clauses(
        self, maven_module: MavenModule, work_dir: Path, out_dir: Path
    ) -> Set[ThrowsClause]:
        throws_clauses: Set[ThrowsClause] = set()
        seen_methods: Set[str] = set()

        # scan and instrument all classes
        # for main subdir: find & instrument methods with throws clause
        main_config = {
            "classroot": str(maven_module.main_classpath),
            "outPath": str(work_dir / "main-out.json"),
            "tcMethodsLogPath": str(work_dir / "tc-methods-logs.txt"),
            "exceptionLogPath": str(work_dir / "exception-logs.txt"),
            "debugPath": str(out_dir / "debug.txt"),
            "modify": True,
        }
        main_config_path = work_dir / "main-config.json"
        su.io.dump(main_config_path, main_config)
        su.bash.run(
            f"java -cp {Tool.core_jar} org.etestgen.core.ClassMainInstrumentor {main_config_path}",
            0,
        )

        # load the throws clauses
        for data in su.io.load(work_dir / "main-out.json"):
            if data["method"] in seen_methods:
                existing_exceptions = [
                    tc.exception for tc in throws_clauses if tc.method == data["method"]
                ]
                logger.warning(
                    f"Duplicate method {data['method']}, existing {existing_exceptions}, new {data['exceptions']}"
                )
            for exception in data["exceptions"]:
                exception = exception.replace("/", ".")
                throws_clauses.add(ThrowsClause(data["method"], exception))
            seen_methods.add(data["method"])

        # save collected data
        su.io.dump(out_dir / "throws_clauses.jsonl", throws_clauses)
        return throws_clauses

    def gen_randoop_tests(
        self, maven_module: MavenModule, work_dir: Path, out_dir: Path
    ):
        # figure out the classpath
        classpath_parts = [maven_module.main_classpath, Tool.junit4_classpath]
        for dep in maven_module.dependency_classpath.split(os.pathsep):
            if "junit" in dep or "hamcrest" in dep:
                continue
            classpath_parts.append(dep)
        classpath = os.pathsep.join(classpath_parts)

        # figure out the class to test
        classlist = []
        for cf in Path(maven_module.main_classpath).rglob("*.class"):
            # TODO: scan using ASM, and filter out the cases that depend on something other than exec_classpath
            if cf.name == "package-info.class":
                continue
            cname = str(cf.relative_to(maven_module.main_classpath))[:-6].replace(
                "/", "."
            )
            classlist.append(cname)
        su.io.dump(work_dir / "randoop.classlist.txt", classlist, su.io.Fmt.txtList)

        rr = su.bash.run(
            f"""java\
            -Xmx32g\
            -classpath {classpath}:{Tool.randoop_jar}\
            randoop.main.Main\
            gentests\
            --classlist={work_dir}/randoop.classlist.txt\
            --time-limit={self.config['randoop_time_limit']}\
            --generated-limit={self.config['randoop_num_limit']}\
            --junit-output-dir={work_dir}/randoop\
            --npe-on-null-input=INVALID\
            --npe-on-non-null-input=INVALID\
            --testsperfile=100\
            --usethreads=true
            """,
            0,
        )
        su.io.dump(out_dir / "randoop.stdout.txt", rr.stdout, su.io.Fmt.txt)
        su.io.dump(out_dir / "randoop.stderr.txt", rr.stderr, su.io.Fmt.txt)

        randoop_config = {
            "classpath": classpath,
            "mainSrcRoot": maven_module.main_srcpath,
            "testSrcRoot": str(work_dir / "randoop"),
            "outPath": str(work_dir / "randoop.out.json"),
            "debugPath": str(out_dir / "debug.txt"),
            "noCContext": True,  # Randoop generated tests don't require class context, except for the "debug" guard we're removing soon
        }
        randoop_config_path = work_dir / "randoop.config.json"
        su.io.dump(randoop_config_path, randoop_config)
        su.bash.run(
            f"java -Xmx32g -cp {Tool.core_jar} org.etestgen.core.SrcTestScanner {randoop_config_path}",
            0,
        )

        tests = su.io.load(work_dir / "randoop.out.json", clz=List[TestMethod])
        # remove the "debug" guard at the beginning of each test
        for test in tests:
            test.code = test.code.replace("if (debug)", "if (false)")
        su.io.dump(out_dir / "randoop.tests.jsonl", tests)

    def get_potential_mnames(self, throws_clauses: Set[ThrowsClause]) -> List[str]:
        """Get the names of the methods with declared exceptions, which are for identifying the tests that may cover those methods."""
        potential_mnames = set()
        for tc in throws_clauses:
            cname, mname, _ = tc.method.split("#", 2)
            if mname == "<init>":
                mname = cname.split("/")[-1]
            potential_mnames.add(mname)
        return list(potential_mnames)

    test_placeholder = "/*TEST PLACEHOLDER*/"

    def run_netests_compute_coverage(
        self,
        maven_module: MavenModule,
        kind: str,
        work_dir: Path,
        out_dir: Path,
    ):
        """
        Run normal (non exceptional-behavior) tests and compute the coverage.
        """

        if kind == "manual":
            classpath = os.pathsep.join(
                [
                    maven_module.main_classpath,
                    maven_module.test_classpath,
                    maven_module.dependency_classpath,
                ]
            )
        else:
            classpath = os.pathsep.join(
                [Tool.junit4_classpath, maven_module.exec_classpath]
            )

        tests: List[TestMethod] = su.io.load(
            out_dir / f"{kind}.tests.jsonl", clz=TestMethod
        )

        tc_method_log = work_dir / "tc-methods-logs.txt"
        ex_log = work_dir / "exception-logs.txt"
        tests_traces = []
        for test_i, test in enumerate(tests):
            if test.pattern is not None:
                # skip etests
                continue

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

            test_file_content = (
                f"public void adhoc_test() throws Exception {test.code}\n"
            )
            test_file_content = f"@org.junit.Test\n" + test_file_content
            test_file_content = ccontext.replace(
                self.test_placeholder, test_file_content
            )
            su.io.dump(test_path, test_file_content, su.io.Fmt.txt)
            test_log_name = ""
            test_log_name = test_i

            with su.io.cd(run_path):
                # compile the test
                rr = su.bash.run(f"javac -cp {classpath} {test_path}")

                if rr.returncode != 0:
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.java",
                        test_file_content,
                        su.io.Fmt.txt,
                    )
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.log",
                        "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                        su.io.Fmt.txt,
                    )
                    continue

                # delete old log
                su.io.rm(tc_method_log)
                su.io.rm(ex_log)

                # run the test
                try:
                    rr = su.bash.run(
                        f"java -cp .:{Tool.rt_jar}:{classpath} -ea org.junit.runner.JUnitCore {test_name}",
                        timeout=self.config["test_timeout"],
                    )
                except subprocess.TimeoutExpired:
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.java",
                        test_file_content,
                        su.io.Fmt.txt,
                    )
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.log",
                        "TIMEOUT",
                        su.io.Fmt.txt,
                    )

                    continue

                if rr.returncode != 0:
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.java",
                        test_file_content,
                        su.io.Fmt.txt,
                    )
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.log",
                        "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                        su.io.Fmt.txt,
                    )
                    continue
                # read the logs (log the beginning of each non-abstract method within the project)
                if ex_log.exists():
                    # etest for unchecked exception
                    called_methods: List[str] = su.io.load(ex_log, su.io.Fmt.txtList)
                    called_methods_stack_trace = process_logged_stacktraces(
                        called_methods
                    )
                    coverage = Coverage(
                        test_i,
                        [],
                        called_methods_stack_trace,
                        ccontext.replace(self.test_placeholder, ""),
                    )
                    tests_traces.append(coverage)
        # save collected data
        su.io.dump(out_dir / f"{kind}.netest-call-trace.jsonl", tests_traces)

    def run_etests_compute_coverage(
        self,
        maven_module: MavenModule,
        kind: str,
        work_dir: Path,
        out_dir: Path,
        potential_mnames: List[str],
    ):
        """
        Run the exceptional-behavior tests and compute the coverage.
        """

        if kind == "manual":
            classpath = os.pathsep.join(
                [
                    maven_module.main_classpath,
                    maven_module.test_classpath,
                    maven_module.dependency_classpath,
                ]
            )
        else:
            classpath = os.pathsep.join(
                [Tool.junit4_classpath, maven_module.exec_classpath]
            )

        tests: List[TestMethod] = su.io.load(
            out_dir / f"{kind}.tests.jsonl", clz=TestMethod
        )

        # coverages of the tests expecting exceptions
        etest_coverages: List[Coverage] = []
        # etest not able to run
        etest_error: List[int] = []
        # etest coverage unchecked exception
        etest_unchecked_exception: List[int] = []
        # coverages of other tests
        other_coverages: List[Coverage] = []
        # etests expecting exceptions but not covering any methods (based on my algorithm)
        etest_no_coverage: List[Coverage] = []

        tc_method_log = work_dir / "tc-methods-logs.txt"
        ex_log = work_dir / "exception-logs.txt"
        for test_i, test in enumerate(tests):
            if test.pattern is None:
                # check if it has the potential to cover a method with declared exceptions
                for mname in potential_mnames:
                    if mname in test.code:
                        is_etest = False
                        break
                else:
                    # skip running tests that don't appear to cover any methods with declared exceptions
                    continue
            else:
                # tests expecting an exception
                is_etest = True
            if not is_etest:
                continue
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

            test_file_content = (
                f"public void adhoc_test() throws Exception {test.code}\n"
            )
            if is_etest and (test.pattern == "@Test(expected)"):
                test_file_content = (
                    f"@org.junit.Test(expected={test.exception}.class)\n"
                    + test_file_content
                )
            else:
                test_file_content = f"@org.junit.Test\n" + test_file_content
            test_file_content = ccontext.replace(
                self.test_placeholder, test_file_content
            )
            su.io.dump(test_path, test_file_content, su.io.Fmt.txt)
            test_log_name = ""
            if is_etest:
                test_log_name = f"etest-{test_i}"
            else:
                test_log_name = test_i
            with su.io.cd(run_path):
                # compile the test
                rr = su.bash.run(f"javac -cp {classpath} {test_path}")

                if rr.returncode != 0:
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.java",
                        test_file_content,
                        su.io.Fmt.txt,
                    )
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.log",
                        "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                        su.io.Fmt.txt,
                    )
                    if is_etest:
                        etest_error.append(test_i)
                    continue

                # delete old log
                su.io.rm(tc_method_log)
                su.io.rm(ex_log)

                # run the test
                try:
                    rr = su.bash.run(
                        f"java -cp .:{Tool.rt_jar}:{classpath} -ea org.junit.runner.JUnitCore {test_name}",
                        timeout=self.config["test_timeout"],
                    )
                except subprocess.TimeoutExpired:
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.java",
                        test_file_content,
                        su.io.Fmt.txt,
                    )
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.log",
                        "TIMEOUT",
                        su.io.Fmt.txt,
                    )
                    if is_etest:
                        etest_error.append(test_i)
                    continue

                if rr.returncode != 0:
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.java",
                        test_file_content,
                        su.io.Fmt.txt,
                    )
                    su.io.dump(
                        out_dir / f"{kind}.error" / f"{test_log_name}.log",
                        "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                        su.io.Fmt.txt,
                    )
                    if is_etest:
                        etest_error.append(test_i)
                    continue
                # read log
                if tc_method_log.exists():
                    lines: List[str] = su.io.load(tc_method_log, su.io.Fmt.txtList)
                    if ex_log.exists():
                        throwed_exs: List[str] = su.io.load(ex_log, su.io.Fmt.txtList)
                    else:
                        throwed_exs = []
                    # process logs
                    called_methods_stacktraces = process_logged_stacktraces(lines)
                    throw_exs_methods_stacktraces = process_logged_stacktraces(
                        throwed_exs
                    )
                    coverage = Coverage(
                        test_i,
                        called_methods_stacktraces,
                        throw_exs_methods_stacktraces,
                        ccontext.replace(self.test_placeholder, ""),
                    )
                    if is_etest:
                        if len(coverage.methods) == 0:
                            # ok this is an unchecked exception
                            etest_unchecked_exception.append(coverage)
                        else:
                            etest_coverages.append(coverage)
                    else:
                        other_coverages.append(coverage)  # ne test cover tc method
                elif is_etest and ex_log.exists():
                    # etest for unchecked exception
                    throwed_exs: List[str] = su.io.load(ex_log, su.io.Fmt.txtList)
                    throw_exs_methods_stacktraces = process_logged_stacktraces(
                        throwed_exs
                    )
                    coverage = Coverage(
                        test_i,
                        [],
                        throw_exs_methods_stacktraces,
                        ccontext.replace(self.test_placeholder, ""),
                    )
                    etest_unchecked_exception.append(coverage)
                elif is_etest:
                    coverage = Coverage(
                        test_i,
                        [],
                        [],
                        ccontext.replace(self.test_placeholder, ""),
                    )
                    etest_no_coverage.append(coverage)

        # save collected data
        su.io.dump(out_dir / f"{kind}.etest-tc-coverages.jsonl", etest_coverages)
        su.io.dump(
            out_dir / f"{kind}.etest-unchecked-exception.jsonl",
            etest_unchecked_exception,
        )
        su.io.dump(out_dir / f"{kind}.netest-tc-coverages.jsonl", other_coverages)
        su.io.dump(out_dir / f"{kind}.etest-error.jsonl", etest_error)
        su.io.dump(out_dir / f"{kind}.etest-no-coverage.jsonl", etest_no_coverage)

    RE_INNER_CLASS = re.compile(r"\$[0-9]+")

    def compute_normal_coverage(
        self,
        out_dir: su.arg.RPath,
        repos_file: su.arg.RPath = Macros.work_dir
        / "repos"
        / "filtered"
        / "repos.json",
        project_names: Optional[Union[List[str], str]] = None,
        skip_project_names: Optional[List[str]] = None,
        overwrite: bool = True,
    ):
        """
        Find normal tests' coverage given the projects list.
        """

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
        if skip_project_names is not None:
            new_projects = [
                p for p in projects if p.full_name not in skip_project_names
            ]
            skip = len(projects) - len(new_projects)
            logger.info(f"Skipped {skip} projects")
            projects = new_projects
        pbar = tqdm(total=len(projects))
        for p in projects:
            pbar.set_description(
                f"Processing {p.full_name} (+{success} -{fail} s{skip})"
            )
            try:
                with su.TimeUtils.time_limit(self.config["project_time_limit"]):
                    self.compute_netest_coverage(p, out_dir / p.full_name)
                    success += 1
            except KeyboardInterrupt:
                raise
            except:
                logger.warning(f"Failed to process {p.full_name}")
                fail += 1
                su.io.dump(
                    out_dir / p.full_name / "netest-collection-error.txt",
                    traceback.format_exc(),
                )
            finally:
                pbar.update(1)
        pbar.set_description(f"Finished (+{success} -{fail} s{skip})")
        pbar.close()

    def compute(
        self,
        out_dir: su.arg.RPath,
        repos_file: su.arg.RPath = Macros.work_dir
        / "repos"
        / "filtered"
        / "repos.json",
        project_names: Optional[List[str]] = None,
        skip_project_names: Optional[List[str]] = None,
        overwrite: bool = True,
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
        if skip_project_names is not None:
            new_projects = [
                p for p in projects if p.full_name not in skip_project_names
            ]
            skip = len(projects) - len(new_projects)
            logger.info(f"Skipped {skip} projects")
            projects = new_projects

        pbar = tqdm(total=len(projects))
        for p in projects:
            pbar.set_description(
                f"Processing {p.full_name} (+{success} -{fail} s{skip})"
            )
            try:
                su.io.mkdir(out_dir / p.full_name, fresh=overwrite)
                with su.TimeUtils.time_limit(self.config["project_time_limit"]):
                    self.compute_project(p, out_dir / p.full_name)
                    success += 1
            except KeyboardInterrupt:
                raise
            except:
                logger.warning(f"Failed to process {p.full_name}")
                fail += 1
                su.io.dump(out_dir / p.full_name / "error.txt", traceback.format_exc())
            finally:
                pbar.update(1)
        pbar.set_description(f"Finished (+{success} -{fail} s{skip})")
        pbar.close()


def check_throw_stmt_in_method(ms: MethodStructure, ttype: str) -> bool:
    """
    Return true if there is the exception ttype is thrown in the method ms.
    """

    if not ms.bytecode:
        return False

    if Insn("ATHROW", []) not in ms.bytecode.get_ordered_insns():
        return False

    for ast in ms.ast.traverse():
        if ast.ast_type == "ThrowStmt":
            for child in ast.children:
                if child.ast_type == "ObjectCreationExpr":
                    for gchild in child.children:
                        if (
                            gchild.ast_type == "ClassOrInterfaceType"
                            and gchild.raw_code == ttype.split(".")[-1]
                        ):
                            return True
    return False


def get_callee_methods(
    call_graph: CallGraph, all_method_list: List[MethodStructure], caller_id: int
):
    cns = call_graph.get_edges_from(caller_id)
    callee_methods = []
    for nid, _ in cns:
        callee_methods.append(all_method_list[nid])
    return callee_methods


def dfs(
    call_graph: CallGraph,
    cur_ms: MethodStructure,
    exception: str,
    call_stack: List,
    result_stacks: List,
    all_methods_list: List[MethodStructure],
):
    call_stack.append(cur_ms)

    if check_throw_stmt_in_method(cur_ms, exception):
        result_stacks.append(deepcopy(call_stack))
    else:
        callee_methods = get_callee_methods(call_graph, all_methods_list, cur_ms.id)
        for ms in callee_methods:
            if exception not in ms.ttypes:
                continue
            dfs(
                call_graph,
                ms,
                exception,
                call_stack,
                result_stacks,
                all_methods_list,
            )
    call_stack.pop()


def find_call_stacks_with_throw(
    method_struct: MethodStructure,
    p_cg: CallGraph,
    all_methods_list: List[MethodStructure],
):
    """find all call stacks starting from method_struct that ends with a method throws the exception in method_struct.ttypes"""
    result_stacks = []
    for ttype in method_struct.ttypes:
        call_stack = []
        dfs(p_cg, method_struct, ttype, call_stack, result_stacks, all_methods_list)
    return result_stacks


def process_logged_stacktraces(raw_stack_traces: List[str]) -> List[StackTrace]:
    """
    Process logged stacktraces.

    Look at the traced methods (either with throw clause or with throw stmt) until test method itself.
    Note that getStackTrace method calls and LogHelper methods are ignored.

    Returns:
        List[StackTrace]: a list of stack traces data structure

    """

    stack_trace_list = []
    for method_call in raw_stack_traces:
        stack_trace = StackTrace()
        try:
            stack_trace.stack_depth = int(method_call.split("@@")[0])
            stack_trace.logged_method = method_call.split("@@")[1]
            traced_method_list = []
            raw_traced_methods: List[str] = method_call.split("@@")[2].split("##")
        except:
            continue

        test_method_exists = False
        for raw_traced_method in raw_traced_methods:
            if raw_traced_method == "":
                continue
            if (
                raw_traced_method.split("#")[0] == "java.lang.Thread"
                and raw_traced_method.split("#")[1] == "getStackTrace"
            ):
                continue
            if (
                raw_traced_method.split("#")[0] == "org.etestgen.rt.LogHelper"
                and raw_traced_method.split("#")[1] == "log"
            ):
                continue
            try:
                if raw_traced_method.split("#")[1] == "adhoc_test":
                    test_method_exists = True
                    break
            except IndexError:
                # logger.warning(f"Index error for traced method {raw_traced_method}")
                continue
            try:
                traced_method = TracedMethod()
                traced_method.class_name = raw_traced_method.split("#")[0]
                traced_method.method_name = raw_traced_method.split("#")[1]
                traced_method.called_line_number = int(raw_traced_method.split("#")[2])
            except IndexError:
                continue
            except:
                continue
            traced_method_list.append(traced_method)
        # endfor

        if test_method_exists:
            stack_trace.traced_methods = traced_method_list
            stack_trace_list.append(stack_trace)
    return stack_trace_list


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.WARNING)
    CLI(ThrowsCoverageComputer, as_positional=False)
