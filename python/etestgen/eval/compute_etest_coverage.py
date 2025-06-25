import dataclasses
import os
import re
import subprocess
import traceback
from pathlib import Path
from typing import List, Optional, Set, Union, Any, Tuple
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
from etestgen.eval.compute_throws_coverage import (
    scan_manual_tests,
    TestMethod,
    ThrowsClause,
    TracedMethod,
)

logger = su.log.get_logger(__name__)
test_placeholder = "/*TEST PLACEHOLDER*/"


@dataclasses.dataclass
class Coverage:
    test_i: int
    methods: List[TracedMethod] = dataclasses.field(default_factory=list)


class EtestCoverageComputer:
    DOWNLOADS_DIR = Macros.downloads_dir

    def __init__(
        self,
        test_timeout: int = 15,
        project_time_limit: int = 1800,
        randoop_time_limit: int = 120,
        randoop_num_limit: int = 100000,
    ):
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.raw_data_dir = Macros.work_dir / "raw-data"

    def compute_coverage_data(
        self,
        out_dir: su.arg.RPath = Macros.work_dir / "coverage-new",
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

    def compute_project(self, project: Project, out_dir: Path):
        # clone, checkout, clean
        project.clone(self.DOWNLOADS_DIR, exists="remove")
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
            # if this module should be skipped
            skip = self.skip_maven_module(maven_module, module_i)
            if skip:
                skipped.append(skip)
                continue

            # checkout and recompile the whole project
            maven_proj.restore_pom()
            project.checkout(project.data["sha"], forced=True)
            with su.io.cd(project.dir):
                su.bash.run("git clean -ffdx")
            maven_proj.backup_pom()
            # maven_proj.hack_pom_delete_plugin("")
            maven_proj.compile()
            maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")

            # prepare work dir and out dir
            work_dir = su.io.mktmp_dir("etestgen")
            module_out_dir = out_dir / f"{module_i}"
            su.io.dump(module_out_dir / "module.yaml", maven_module)

            # collect manual tests
            try:
                scan_manual_tests(
                    maven_module, work_dir, module_out_dir
                )  # already collected
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
            self.run_etests_compute_coverage(maven_module, work_dir, module_out_dir)

            su.io.rmdir(work_dir)
        if len(skipped) > 0:
            su.io.dump(out_dir / "skipped.jsonl", skipped)

    def skip_maven_module(self, maven_module: Any, module_i: int):
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

    def run_etests_compute_coverage(
        self,
        maven_module: MavenModule,
        work_dir: Path,
        out_dir: Path,
    ):
        """
        Run the exceptional-behavior tests and compute the coverage.
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
        etests_coverage_data = []

        ex_log = work_dir / "stack-trace-logs.txt"
        for test_i, test in enumerate(tests):
            if test.pattern is None:
                # skip running tests that don't appear to cover any methods with declared exceptions
                continue
            test_file_content, test_path, test_name = extract_adhoc_test(work_dir, test)
            su.io.dump(test_path, test_file_content, su.io.Fmt.txt)
            test_log_name = f"etest-{test_i}"

            # run the test
            su.io.rm(ex_log)
            with su.io.cd(work_dir / "run"):
                res = self.run_adhoc_test(
                    test_path,
                    classpath,
                    out_dir,
                    test_log_name,
                    test_name,
                    test_file_content,
                )
                if not res:
                    continue
                # read log
                if ex_log.exists():
                    cov_data = self.analyze_etest_logs(ex_log, test_i)
                    etests_coverage_data.append(cov_data)

        # save collected data
        su.io.dump(out_dir / "manual.etests-coverage.jsonl", etests_coverage_data)

    def run_adhoc_test(
        self,
        test_path: str,
        classpath: str,
        out_dir: Path,
        test_log_name: str,
        test_name: str,
        test_file_content: str,
    ) -> bool:
        """
        Run the adhoc test seperately. if there is no error, return True, otherwise return False.
        """
        # compile the test
        rr = su.bash.run(f"javac -cp {classpath} {test_path}")

        if rr.returncode != 0:
            su.io.dump(
                out_dir / f"manual.error" / f"{test_log_name}.java",
                test_file_content,
                su.io.Fmt.txt,
            )
            su.io.dump(
                out_dir / f"manual.error" / f"{test_log_name}.log",
                "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                su.io.Fmt.txt,
            )
            return False
        try:
            rr = su.bash.run(
                f"java -cp .:{Tool.rt_jar}:{classpath} -ea org.junit.runner.JUnitCore {test_name}",
                timeout=self.config["test_timeout"],
            )
        except subprocess.TimeoutExpired:
            su.io.dump(
                out_dir / f"manual.error" / f"{test_log_name}.java",
                test_file_content,
                su.io.Fmt.txt,
            )
            su.io.dump(
                out_dir / f"manual.error" / f"{test_log_name}.log",
                "TIMEOUT",
                su.io.Fmt.txt,
            )
            return False

        if rr.returncode != 0:
            su.io.dump(
                out_dir / f"manual.error" / f"{test_log_name}.java",
                test_file_content,
                su.io.Fmt.txt,
            )
            su.io.dump(
                out_dir / f"manual.error" / f"{test_log_name}.log",
                "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                su.io.Fmt.txt,
            )
            return False

        return True

    def analyze_etest_logs(self, ex_log: Path, test_i: int) -> Coverage:
        """
        Extract coverage data from logs.
        """
        called_methods = su.io.load(ex_log, su.io.Fmt.txt)
        called_methods_stack_trace = process_logged_stacktraces(called_methods)
        coverage = Coverage(
            test_i,
            called_methods_stack_trace,
        )
        return coverage


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


def extract_adhoc_test(work_dir: Union[Path, str], test: TestMethod):
    # extract the test to an ad-hoc test class
    run_path = work_dir / "run"
    su.io.mkdir(run_path, fresh=True)
    ccontext = test.ccontext
    if test.ccontext is None:
        package = None
        test_name = "adhoc_Test"
        test_path = run_path / f"{test_name}.java"
        ccontext = "public class adhoc_Test { " + test_placeholder + " }"
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
    modified_test, test_file_content = add_try_catch_block(
        test.code, ccontext, work_dir / "stack-trace-logs.txt"
    )

    test_file_content = test_file_content.replace(test_placeholder, modified_test)
    return test_file_content, test_path, test_name


def add_try_catch_block(
    test_body: str, test_file_context: str, log_file_path: Union[Path, str]
) -> Tuple[str, str]:
    # print stack trace code
    print_stack_trace_method = """public static String formatStackTrace(StackTraceElement stackTrace[]) {
        StringBuilder sb = new StringBuilder();
        for (StackTraceElement ste : stackTrace) {
            sb.append(ste.getClassName());
            sb.append("#");
            sb.append(ste.getMethodName());
            sb.append("#");
            sb.append(ste.getLineNumber());
            sb.append("##");
        }
        return sb.toString();
    }"""
    # extra libs
    imported_libs = """import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
"""

    # add try catch block
    try_block = "try {\n"
    catch_block = (
        """} catch (Exception e) {
            StackTraceElement[] stackTrace;
            if (e.getCause() != null) {
                stackTrace = e.getCause().getStackTrace();
            } else {
                stackTrace = e.getStackTrace();
            }
            String stackTraceString = formatStackTrace(stackTrace);
            try {
                Files.write(Paths.get(\""""
        + f"{log_file_path}"
        + """\"), stackTraceString.getBytes(),
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        }
            catch (Exception ex) {
            }
        }
    }
    """
    )
    catch_block += print_stack_trace_method
    assert test_body[0] == "{" and test_body[-1] == "}"
    modified_test = (
        "@Test\n"
        + "public void adhoc_test() throws Exception{\n"
        + try_block
        + test_body[1:-1]
        + catch_block
    )
    # put package statement at the first line
    if test_file_context.startswith("package"):
        package_line = test_file_context.splitlines()[0] + "\n"
        remainining_context = test_file_context.split("\n", 1)[1]
        test_file_context = package_line + imported_libs + remainining_context
    else:
        test_file_context = imported_libs + test_file_context
    return modified_test, test_file_context


def process_logged_stacktraces(raw_stack_traces: str) -> List[TracedMethod]:
    """
    Process logged stacktraces.

    Look at the traced methods (either with throw clause or with throw stmt) until test method itself.
    Note that getStackTrace method calls and LogHelper methods are ignored.

    Returns:
        List[StackTrace]: a list of stack traces data structure

    """

    stack_trace_list = []
    raw_stack_traces = raw_stack_traces.split("##")
    raw_stack_traces = list(filter(lambda x: x != "", raw_stack_traces))

    for stack_trace_frame in raw_stack_traces:
        try:
            class_name = stack_trace_frame.split("#")[0]
            method_name = stack_trace_frame.split("#")[1]
            line_number = stack_trace_frame.split("#")[2]
            if method_name == "adhoc_test" and "adhoc_" in class_name:
                break
            traced_method = TracedMethod(class_name, method_name, line_number)
            stack_trace_list.append(traced_method)
        except IndexError:
            # ignore some special called methods
            pass
    return stack_trace_list


if __name__ == "__main__":
    su.log.setup(Macros.stack_trace_log_file, su.log.WARNING)
    CLI(EtestCoverageComputer, as_positional=False)
