import collections
import dataclasses
import functools
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, overload
import os
from collections import defaultdict

import seutil as su
from jsonargparse import CLI
from seutil.maven import MavenModule, MavenProject
from seutil.project import Project
from tqdm import tqdm

from etestgen.data.data import DataNE2E, TData, DataMUT2E
from etestgen.data.tool import Tool
from etestgen.data.utils import save_dataset, load_dataset
from etestgen.eval.compute_throws_coverage import (
    Coverage,
    TestMethod,
    ThrowsClause,
    ThrowsClauseMethod,
    StackTrace,
    TracedMethod,
)
from etestgen.data.structures import AST, ClassStructure, Consts, MethodStructure, Scope
from etestgen.macros import Macros

import sys

sys.setrecursionlimit(10000)

logger = su.log.get_logger(__name__)


class CodeMappingException(Exception):
    pass


@dataclasses.dataclass
class ExceptionMethod:
    method: str = None
    line_number: int = None
    # the previous 2 lines and the next 2 lines of the thrown statement
    context_radius: int = 2
    context: List[str] = None


@dataclasses.dataclass
class etestContext:
    mut: str = None
    mte: ExceptionMethod = None
    exception_stacktrace: List[TracedMethod] = None


@dataclasses.dataclass
class ResolvedTestsCoverages:
    # tests
    tests: List[TestMethod] = dataclasses.field(default_factory=list)
    # ids of e-tests (exceptional-behavior tests)
    etests: List[int] = dataclasses.field(default_factory=list)

    # etest cover the throws clause
    etest_tc_coverage: List[Coverage] = dataclasses.field(default_factory=list)
    # tc that is covered by etest
    etest_cover_tc: List[int] = dataclasses.field(default_factory=list)

    # netest cover the throws clause
    netest_coverage: List[Coverage] = dataclasses.field(default_factory=list)
    netest_cover_tc: List[int] = dataclasses.field(default_factory=list)
    # etest that cover a method no throw clause
    etest_unk_exception: List[Coverage] = dataclasses.field(default_factory=list)
    etest_uncheck: List[int] = dataclasses.field(default_factory=list)

    # etest unable to see what it covers
    etest_no_coverage: List[Coverage] = dataclasses.field(default_factory=list)
    etest_no_mut: List[int] = dataclasses.field(default_factory=list)
    # throws clause coverage rate
    coverage: float = 0.0
    etests_error: List[int] = dataclasses.field(default_factory=list)

    # the mapping from throws clause ids to the list of e-test ids that cover them
    tc_2_e: Dict[int, Set[int]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set)
    )
    # mut to etest
    mut_2_e: Dict[str, Set[int]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set)
    )
    # mut to netest
    mut_2_ne: Dict[str, Set[int]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set)
    )
    # mut without tc to etest
    unk_2_e: Dict[str, Set[int]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set)
    )

    # exception types covered by e-tests
    covered_types: Set[str] = dataclasses.field(default_factory=set)
    # (unchecked) exception types covered by e-tests (it is not declared)
    unk_covered_types: Set[str] = dataclasses.field(default_factory=set)

    # number of errors ignored during collection
    num_error: int = 0

    @functools.cached_property
    def ne_e_pairs(self) -> List[Tuple[int, int, int]]:
        """
        pairs of ne-test and e-test co-covering the same method, in the format of list of (ne-test id, e-test id, throws clause ids)
        """

        ret = []
        shared_tc_ids = set(self.tc_2_e.keys()) & set(self.covered_by_ne.keys())
        for tc_id in shared_tc_ids:
            for ne_id in self.covered_by_ne[tc_id]:
                for e_id in self.tc_2_e[tc_id]:
                    ret.append((ne_id, e_id, tc_id))
        return ret


def extract_stmts_from_ast(ast: AST) -> List[AST]:
    test_stmts = []
    # collect stmt & cf asts; record their line numbers
    front_lineno = 0
    last_was_stmt = False
    lineno2stmt_i = {}
    front_cf = []
    # read each statement in test body, excluding opening and closing brackets
    for n_out in ast.get_body().children[1:-1]:
        for n in n_out.traverse(lambda x: x.ast_type in Consts.asts_terminal_stmt):
            if n.ast_type in Consts.asts_terminal_stmt:
                # terminal statement
                lineno_beg, lineno_end = n.get_lineno_range()
                if lineno_beg <= front_lineno:
                    raise CodeMappingException("multiple statements on same line")
                front_lineno = lineno_end
                for lineno in range(lineno_beg, lineno_end + 1):
                    lineno2stmt_i[lineno] = len(test_stmts)
                front_cf = []
                test_stmts.append(n)
                last_was_stmt = True
            elif n.is_terminal():
                # control flow tokens in between terminal statements
                lineno_beg, lineno_end = n.get_lineno_range()
                if lineno_beg < front_lineno:
                    raise CodeMappingException("multiple statements on same line")
                elif lineno_beg == front_lineno and last_was_stmt:
                    raise CodeMappingException("multiple statements on same line")
                    # it is ok to have multiple control flow tokens on the same line
                front_lineno = lineno_end
                front_cf.append(n.tok)
                last_was_stmt = False
            # other non-terminals
    return test_stmts


def map_traced_method_to_method_structure(
    called_traces: List[str],
    p_methods_list: List[MethodStructure],
    p_class_list: List[ClassStructure],
    p_class_offset: int,
) -> List[MethodStructure]:
    """
    Map the method collected from stack traces to method structure.

    Return: list of methods called in the traces.
    """
    # map trace to method structure
    called_ms = []
    for callee in called_traces:
        cls_name, method_name, linum = callee
        if (
            "junit" in cls_name
            or "java.util" in cls_name
            or "adhoc_" in cls_name
            or "java.lang" in cls_name
        ):
            continue
        candidate_matched_methods = []
        for ms in p_methods_list:
            if not ms.code:
                continue
            ms_class_name = p_class_list[ms.clz - p_class_offset].name
            ms_name = ms.name
            if (cls_name == ms_class_name) and (ms_name == method_name):
                ms.cls_name = ms_class_name.replace(".", "/")
                ms.key = cls_name + "#" + ms.namedesc
                candidate_matched_methods.append(ms)
        #
        if len(candidate_matched_methods) == 1:
            called_ms.append(candidate_matched_methods[0])
        elif len(candidate_matched_methods) > 1:
            # find the one that matches the line number
            for ms in candidate_matched_methods:
                if ms.start_line <= linum <= ms.end_line:
                    called_ms.append(ms)
                    break
        #
    return called_ms


@dataclasses.dataclass
class ModuleData:
    project: str
    module: str
    module_i: int
    throws_clauses: List[ThrowsClause] = None
    methods: List[TestMethod] = None
    declared_types: List[str] = None
    mdata: Optional[ResolvedTestsCoverages] = None
    rdata: Optional[ResolvedTestsCoverages] = None
    udata: Optional[ResolvedTestsCoverages] = None
    report: str = dataclasses.field(default_factory=str)


def extract_methods_traces_from_logs(
    logged_traces: List[StackTrace],
) -> List[Tuple[str]]:
    """
    Extract a list methods from the logged traces of the last invoked method while running the tests.
    """

    last_logged_trace = logged_traces[-1].traced_methods[::-1]

    called_method_traces = []
    for tm in last_logged_trace:
        called_method_traces.append(
            (tm.class_name, tm.method_name, tm.called_line_number)
        )
    return called_method_traces


class ThrowsCoverageAnalyzer:
    """Scripts for analyzing the coverage results."""

    maven_src_path = "src/main/java"

    def __init__(
        self,
        coverage_dir: su.arg.RPath = Macros.work_dir / "coverage",
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
        self.raw_data_dir = Macros.work_dir / "raw-data"

    def analyze(
        self,
        out_dir: su.arg.RPath = Macros.results_dir / "coverage",
        error_threshold: float = 0.5,
        target_project_name: List[str] = None,
    ):
        su.io.mkdir(out_dir)
        records_error = []
        records_summary = []
        data_list = []
        all_projects = list(self.coverage_dir.iterdir())
        runnable_projects = [p.name for p in list(self.raw_data_dir.iterdir())]
        for project_dir in tqdm(all_projects, desc="Analyzing"):
            pname = project_dir.name
            # ignore projects with errors
            if pname not in runnable_projects:
                logger.info(f"Skipping {pname} not in runnable list")
                records_error.append(
                    {
                        "project": pname,
                        "error": "Not runnable",
                    }
                )
                continue
            if pname not in self.indexed_projects:
                logger.info(f"Skipping {pname} not in repos list")
                continue
            if target_project_name and pname not in target_project_name:
                continue

            if (project_dir / "error.txt").exists():
                records_error.append(
                    {
                        "project": pname,
                        "error": su.io.load(project_dir / "error.txt"),
                    }
                )
                continue

            # iterate each module in the project
            for module_dir in project_dir.iterdir():
                if not module_dir.is_dir():
                    continue
                module_i = int(os.path.basename(module_dir))
                maven_module = su.io.load(module_dir / "module.yaml", clz=MavenModule)
                module = maven_module.rel_path
                try:
                    data = self.analyze_module(
                        pname, module, module_i, print_report=False
                    )
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

                # drop data if error rate is too high
                if (
                    data.mdata is not None
                    and len(data.mdata.etests) > 0
                    and data.mdata.num_error / len(data.mdata.tests) > error_threshold
                ):
                    records_error.append(
                        {
                            "project": pname,
                            "module": module,
                            "error": f"Too many errors while collecting manual tests data: {data.mdata.num_error}/{len(data.mdata.tests)}",
                        }
                    )
                    continue

                # record the data
                record_summary = {"project": pname, "module": module}
                record_summary["#throw clauses (exceptions could duplicate)"] = len(
                    data.throws_clauses
                )
                record_summary["#method with throw clauses"] = len(data.methods)
                record_summary["#declared unique exceptions"] = len(data.declared_types)

                # log the coverage summary
                for p in ["m", "r", "u"]:
                    if (pdata := getattr(data, f"{p}data")) is None:
                        continue
                    record_summary[f"{p}:#test"] = len(pdata.tests)
                    record_summary[f"{p}:#etest"] = len(pdata.etests)
                    record_summary[f"{p}:#etests cover undeclared exception"] = len(
                        pdata.etest_uncheck
                    )
                    record_summary[f"{p}:#etests not sure what method it covers"] = len(
                        set(pdata.etest_no_mut)
                    )
                    record_summary[f"{p}:#etest that covered throw clause"] = len(
                        pdata.etest_cover_tc
                    )
                    record_summary[f"{p}:#etest-not-able-to-run"] = len(
                        pdata.etests_error
                    )
                    record_summary[f"{p}:#covered throw clause"] = len(pdata.tc_2_e)
                    record_summary[f"{p}:coverage"] = pdata.coverage
                    # record_summary[f"{p}:#ne-e-pair"] = len(pdata.ne_e_pairs)
                    # record_summary[f"{p}:#uncovered-w-ne"] = len(pdata.uncovered_w_ne)
                    record_summary[f"{p}:#type-covered"] = len(pdata.covered_types)
                    record_summary[f"{p}:#type-unk-covered"] = len(
                        pdata.unk_covered_types
                    )

                records_summary.append(record_summary)
                data_list.append(data)
        # printout some intermediate summary
        logger.info(f"In total collected {len(data_list)} modules data.")
        error_projs_count = len({p["project"] for p in records_error})
        collected_projs_count = len({p["project"] for p in records_summary})
        logger.info(f"Projects collected: {collected_projs_count}")
        logger.info(f"Projects with errors: {error_projs_count}")
        su.io.dump(out_dir / "error.json", records_error, su.io.Fmt.jsonPretty)
        su.io.dump(out_dir / "summary.json", records_summary, su.io.Fmt.jsonPretty)
        su.io.dump(out_dir / "data.pkl", data_list)
        su.io.dump(
            out_dir / "report.md",
            "\n\n".join(d.report for d in data_list),
            su.io.Fmt.txt,
        )

    def analyze_module(
        self,
        project: str,
        module: str = ".",  # the rel_path of the module ("." refers to root)
        module_i: int = 0,
        print_report: bool = True,
        list_limit: Optional[int] = 5,
    ) -> ModuleData:
        """
        Resolve the coverage data for a module in a project
        """

        data = ModuleData(project=project, module=module, module_i=module_i)

        proj_coverage_dir = self.coverage_dir / project
        if not proj_coverage_dir.exists():
            raise FileNotFoundError(f"Project {project} does not exist")

        maven_project = su.io.load(proj_coverage_dir / "maven.yaml", clz=MavenProject)
        modules = [m.rel_path for m in maven_project.modules]
        proj_module_dir = Macros.teco_downloads_dir / project / module

        try:
            module_i = modules.index(module)
        except ValueError:
            raise FileNotFoundError(
                f"Module {module} does not exist in project {project}"
            )

        module_coverage_dir = proj_coverage_dir / str(module_i)
        if not module_coverage_dir.exists():
            raise FileNotFoundError(f"Module {module} was not collected")

        data.report += f"# project {project}, module {module}\n"

        # report #throws
        throws_clauses: List[ThrowsClause] = su.io.load(
            module_coverage_dir / "throws_clauses.jsonl", clz=ThrowsClause
        )  # list of (method name and its thorws clause)
        data.throws_clauses = throws_clauses
        methods = list(sorted(set([t.method for t in throws_clauses])))
        data.methods = methods
        exceptions = list(sorted(set([t.exception for t in throws_clauses])))
        data.declared_types = exceptions
        data.report += f"- #throws: {len(throws_clauses)}\n"
        data.report += f"- #methods: {len(methods)}\n"
        data.report += f"- #types: {len(exceptions)}\n"
        data.report += self.format_list(exceptions, list_limit, 2)

        success_kinds = su.io.load(module_coverage_dir / "success-kinds.json")

        valid_data = []
        # report manual tests
        if "manual" not in success_kinds:
            data.report += "- manual tests: failed to collected\n"
        else:
            mdata = self.resolve_tests_coverages(
                module_coverage_dir, "manual", throws_clauses, proj_module_dir
            )  # manual data
            valid_data.append(mdata)
            data.report += f"- manual tests:\n"
            data.report += f"  - #tests: {len(mdata.tests)}\n"
            data.report += f"  - #etests: {len(mdata.etests)}\n"
            data.report += f"  - #throws covered: {len(mdata.etest_cover_tc)}\n"
            data.report += f"  - coverage: {mdata.coverage:.2%}\n"
            # data.report += f"  - #ne-e-pairs: {len(mdata.ne_e_pairs)}\n"
            data.report += (
                f"  - #throws covered w/ etest: {len(set(mdata.etest_cover_tc))}\n"
            )
            data.report += f"  - #covered exception types: {len(mdata.covered_types)}\n"
            data.report += f"  - #etests covered unknown exceptoin types: {len(mdata.unk_covered_types)}\n"
            data.report += self.format_list(mdata.unk_covered_types, list_limit, 4)
            data.mdata = mdata

        # for each uncovered throws, print randoop tests info
        if print_report:
            print(data.report)

        return data

    @classmethod
    def resolve_tests_coverages(
        cls,
        dir: Path,
        kind: str,
        throws_clauses: List[ThrowsClause],
        proj_module_dir: Path,
    ) -> ResolvedTestsCoverages:
        """
        Loads and resolves the tests and coverages for the given kind.
        """
        ret = ResolvedTestsCoverages()

        # load tests and raw coverages data
        ret.tests = su.io.load(dir / f"{kind}.tests.jsonl", clz=TestMethod)
        ret.etests = [i for i, t in enumerate(ret.tests) if t.pattern is not None]
        ret.etest_tc_coverage = su.io.load(
            dir / f"{kind}.etest-tc-coverages.jsonl", clz=Coverage
        )
        netest_coverage = []
        # ret.netest_coverage = su.io.load(
        #     dir / f"{kind}.netest-call-trace.jsonl", clz=Coverage
        # )
        ret.etest_unk_exception = su.io.load(
            dir / f"{kind}.etest-unchecked-exception.jsonl", clz=Coverage
        )
        ret.etest_no_coverage = su.io.load(
            dir / f"{kind}.etest-no-coverage.jsonl", clz=Coverage
        )
        ret.etests_error = su.io.load(dir / f"{kind}.etest-error.jsonl")
        logger.info(
            f"loaded {len(ret.etests)} etests, {len(ret.etest_tc_coverage) + len(ret.etest_unk_exception)} considered"
        )
        # Consider both checked and unchecked etests
        etests = ret.etest_tc_coverage + ret.etest_unk_exception
        # compute covered throws and types
        indexed_tcs = {tc: i for i, tc in enumerate(throws_clauses)}
        method2tc = collections.defaultdict(list)
        for tc_i, tc in enumerate(throws_clauses):
            method2tc[tc.method].append(tc_i)
        for etest in etests:
            t = ret.tests[etest.test_i]
            t.context = etest.test_context
            if len(etest.methods) == 0 and len(etest.mte) > 0:
                mut: str = cls.extract_mut(etest.mte, [])
            elif len(etest.mte) == 0:
                mut = ""
            else:
                mut: str = cls.extract_mut(etest.methods, etest.mte)
            tmp_tc = ThrowsClause(mut, t.exception)
            tmp_tc1 = ThrowsClause(mut, t.exception, True)
            if (tc_i := indexed_tcs.get(tmp_tc)) is not None or (
                tc_i := indexed_tcs.get(tmp_tc1)
            ) is not None:
                ret.tc_2_e[tc_i].add(
                    etest.test_i
                )  # dict {throw_clause_index, test_index}
                ret.etest_cover_tc.append(etest.test_i)
                ret.covered_types.add(t.exception)
            else:
                # etest that cover a exception that is not declared
                if len(etest.mte) > 0:
                    mut: str = cls.extract_mut(etest.mte, [])
                    ret.etest_uncheck.append(etest.test_i)
                    ret.unk_covered_types.add(t.exception)
                    ret.unk_2_e[mut].add(etest.test_i)
                else:
                    ret.etest_no_mut.append(etest.test_i)

        for ne_tc in ret.netest_coverage:
            ret.netest_cover_tc.append(ne_tc.test_i)
        for etnc in ret.etest_no_coverage:
            ret.etest_no_mut.append(etnc.test_i)
        # compute coverage rate
        covered_tc = set(ret.tc_2_e.keys())
        if len(covered_tc) == 0:
            ret.coverage = 0.0
        else:
            ret.coverage = len(covered_tc) / len(throws_clauses)

        # load errors ignored
        if (dir / f"{kind}.error").exists():
            ret.num_error = len(list((dir / f"{kind}.error").glob("*.log")))
        return ret

    @classmethod
    def extract_mut(
        cls,
        tracedMethods: List[StackTrace],
        methods_throw_exceptions: List[StackTrace],
        src_code_methods: dict = None,
    ) -> str:
        """
        Extract the MUT.

        The method under test should be the last called method in etest;
        We log every called methods, and find the logged one that has the "last" "shortest" stack trace.
        """

        # extract the method under test
        mut = ""
        min_depth = float("-inf")

        synthetic_methods = []

        if len(tracedMethods) > 0:
            for traced_method in tracedMethods:
                if src_code_methods and (
                    traced_method.logged_method not in src_code_methods
                ):
                    synthetic_methods.append(traced_method.traced_methods[0])
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
                if stack_depth <= min_depth or min_depth == float("-inf"):
                    min_depth = stack_depth
                    mut = traced_method.logged_method
                assert stack_depth >= min_depth
            # endfor
        else:
            raise RuntimeError("No traced method found")

        return mut

    @classmethod
    def get_context_around(
        cls, file_path: Path, line_number: int, radius: int
    ) -> List[str]:
        """Get the context around the line number, 2 lines above and 2 lines below. (can change later)"""
        with open(file_path, "r") as f:
            lines = f.readlines()
        context = lines[max(0, line_number - radius) : line_number + radius + 1]
        return context

    @classmethod
    def format_list(
        cls, l: List[str], limit: Optional[int] = None, indentation: int = 0
    ) -> str:
        s = ""
        pc = 0
        for e in l:
            s += " " * indentation + f"- {e}\n"
            pc += 1
            if limit is not None and pc >= limit:
                if len(l) != limit:
                    s += " " * indentation + f"- ... and {len(l) - pc} more\n"
                break
        return s

    def print_uncovered_module(
        self,
        project: str,
        module: str = ".",
        list_limit: Optional[int] = 10,
        results_dir: su.arg.RPath = Macros.results_dir / "coverage",
    ):
        data_list: List[ModuleData] = su.io.load(results_dir / "data.pkl")
        data = next(
            (d for d in data_list if d.project == project and d.module == module), None
        )
        if data is None:
            raise RuntimeError(f"no data found for {project} {module}")

        report = ""
        report += f"# project: {project}, module: {module}\n"
        report += f"- sha: {self.indexed_projects[project].data['sha']}\n"
        report += f"- #T: {len(data.mdata.tests)} ({len(data.mdata.etests)} T-E, {len(data.mdata.tests) - len(data.mdata.etests)} T-NE)\n"
        report += f"- #E: {len(data.throws_clauses)}\n"
        report += (
            f"  - covered: {len(data.mdata.covered)} ({data.mdata.coverage:.2%})\n"
        )
        report += f"  - uncovered w/ T-NE: {len(data.mdata.uncovered_w_ne)}\n"

        for case_i, (tc_id, ne_ids) in enumerate(data.mdata.uncovered_w_ne.items()):
            report += "\n\n"
            report += f"## {case_i}\n\n"
            tc = data.throws_clauses[tc_id]
            report += f"- method: {tc.method}\n"
            report += f"- exception: {tc.exception}\n"
            report += f"- T-NE covering this method:\n"
            report += self.format_list(
                [
                    data.mdata.tests[i].cname + "#" + data.mdata.tests[i].mname
                    for i in ne_ids
                ],
                list_limit,
                2,
            )

        print(report)

    def print_covered_module(
        self,
        project: str,
        module: str = ".",
        list_limit: Optional[int] = 10,
        results_dir: su.arg.RPath = Macros.results_dir / "coverage",
    ):
        data_list: List[ModuleData] = su.io.load(results_dir / "data.pkl")
        data = next(
            (d for d in data_list if d.project == project and d.module == module), None
        )
        if data is None:
            raise RuntimeError(f"no data found for {project} {module}")

        report = ""
        report += f"# project: {project}, module: {module}\n"
        report += f"- sha: {self.indexed_projects[project].data['sha']}\n"
        report += f"- #T: {len(data.mdata.tests)} ({len(data.mdata.etests)} T-E, {len(data.mdata.tests) - len(data.mdata.etests)} T-NE)\n"
        report += f"- #E: {len(data.throws_clauses)}\n"
        report += (
            f"  - covered: {len(data.mdata.covered)} ({data.mdata.coverage:.2%})\n"
        )
        report += f"  - uncovered w/ T-NE: {len(data.mdata.uncovered_w_ne)}\n"

        for case_i, (tc_id, e_ids) in enumerate(data.mdata.tc_2_e.items()):
            report += "\n\n"
            report += f"## {case_i}  ({len(e_ids)} test(s))\n\n"
            tc = data.throws_clauses[tc_id]
            report += f"- method: {tc.method}\n"
            report += f"- exception: {tc.exception}\n"
            report += f"- T-E covering this method:\n"
            report += self.format_list(
                [
                    data.mdata.tests[i].cname + "#" + data.mdata.tests[i].mname
                    for i in e_ids
                ],
                list_limit,
                2,
            )
            if tc_id in data.mdata.covered_by_ne:
                report += f"- T-NE covering this method:\n"
                report += self.format_list(
                    [
                        data.mdata.tests[i].cname + "#" + data.mdata.tests[i].mname
                        for i in data.mdata.covered_by_ne[tc_id]
                    ],
                    list_limit,
                    2,
                )
            else:
                report += f"- T-NE covering this method: None\n"

        print(report)

    def collect_sign_stmts(self, data: TData, test_m: TestMethod) -> dict:
        # collect sign ast
        data.test_sign = test_m.ast.get_sign()
        try:
            data.test_stmts = extract_stmts_from_ast(test_m.ast)
        except CodeMappingException:
            data.test_stmts = []
        return data

    # TODO: remove this later
    def test_stack_trace(self):
        test_st_data = su.io.load("../collector/stacktrace-test/test-st-config.json")
        file_name = "./conditions.txt"
        list_argument = []
        test_st_data = test_st_data[-1]
        stack_test_file = "../collector/stacktrace-test/test-st-config.json"
        for mt, linum in zip(
            test_st_data["methodStrings"], test_st_data["lineNumbers"]
        ):
            list_argument.append(f"{mt}@@{linum}")
        argument = "**".join(list_argument)
        Tool.ensure_tool_versions()
        Tool.require_compiled()
        ps = su.bash.run(
            f"java -cp {Tool.core_jar} org.etestgen.core.ConditionCollector '{stack_test_file}' {file_name}",
            0,
        )
        print(ps.stdout)

    def extract_mut2e_data(
        self,
        starting_index: int = 0,
        results_dir: su.arg.RPath = Macros.results_dir / "coverage",
        data_dir: su.arg.RPath = Macros.work_dir / "data" / "mut2e",
        downloads_dir: su.arg.RPath = Macros.downloads_dir,
    ):
        config = {
            k: v
            for k, v in locals().items()
            if k not in {"self", "results_dir", "data_dir", "downloads_dir"}
        }
        su.io.dump(data_dir / "config.json", config, su.io.Fmt.jsonPretty)

        Tool.ensure_tool_versions()
        Tool.require_compiled()

        raw_dataset: List[ModuleData] = su.io.load(results_dir / "data.pkl")
        temp_dir = su.io.mktmp_dir("etestgen")

        filter_counter = {
            "no_mut": 0,
            "no_etest": 0,
            "time_out": 0,
            "keyboard_interrupt": 0,
            "other_error": 0,
        }
        proj_error_logs = {}

        data_id = 0
        raw_dataset = raw_dataset[starting_index:]
        pbar = tqdm(total=len(raw_dataset))
        success = 0
        skip = 0
        error = 0

        # special projects to look at
        for module_data in raw_dataset:
            pbar.set_description(f"total {data_id} (+{success} -{error} x{skip})")
            try:
                cov_data = module_data.mdata
                with su.TimeUtils.time_limit(1800):
                    if len(cov_data.etests) == 0:
                        filter_counter["no_etest"] += 1
                        proj_error_logs[
                            f"{module_data.project}.{module_data.module}"
                        ] = "No etest for this project."
                        skip += 1
                        continue
                    if (
                        len(cov_data.etest_tc_coverage)
                        + len(cov_data.etest_unk_exception)
                        == 0
                    ):
                        skip += 1
                        proj_error_logs[
                            f"{module_data.project}.{module_data.module}"
                        ] = "No MUT found for the etests."
                        filter_counter["no_mut"] += 1
                        continue

                    proj_dataset = []

                    # make sure project is cloned and locate the module
                    project = self.indexed_projects[module_data.project]
                    project.clone(downloads_dir)
                    project.checkout(project.data["sha"], forced=True)
                    maven_project = MavenProject.from_project(project)
                    maven_module = [
                        m
                        for m in maven_project.modules
                        if m.rel_path == module_data.module
                    ][0]

                    # collect all source code in the project
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
                    method2toks = {}
                    method2code = {}
                    for record in records_methods:
                        method2toks[record["method"]] = record["tokens"]
                        method2code[record["method"]] = record["method_node"]
                    p_method_data_dir = (
                        self.raw_data_dir / project.full_name / "joint.method.json"
                    )
                    p_class_data_dir = (
                        self.raw_data_dir / project.full_name / "joint.class.json"
                    )
                    p_methods_list = su.io.load(
                        p_method_data_dir, clz=List[MethodStructure]
                    )
                    p_class_list = su.io.load(
                        p_class_data_dir, clz=List[ClassStructure]
                    )
                    p_class_offset = p_class_list[0].id
                    # setup
                    etest_2_tc = {}
                    for tc_id, eid_set in cov_data.tc_2_e.items():
                        for e_id in eid_set:
                            etest_2_tc[e_id] = tc_id
                    etest_2_cov = {
                        cov.test_i: cov
                        for cov in cov_data.etest_tc_coverage
                        + cov_data.etest_unk_exception
                    }
                    for etest_id in cov_data.etest_cover_tc + cov_data.etest_uncheck:
                        cov_d = etest_2_cov[etest_id]
                        data = DataMUT2E()
                        # project info
                        data.project = project.full_name
                        data.module = module_data.module
                        data.module_i = module_data.module_i
                        # find out method under test (ignore synthetic method)
                        mut = self.extract_mut(cov_d.mte, [], method2code)
                        data.mut_key = mut
                        test_e = cov_data.tests[etest_id]
                        mut_constructor_key = mut.split("#")[0] + "#<init>#"
                        data.constructors = [
                            method2code[mt_key]
                            for mt_key in method2code
                            if mut_constructor_key in mt_key
                        ]
                        try:
                            data.mut_toks = method2toks[mut]
                            data.mut = method2code[mut]
                        except KeyError:
                            logger.warning(
                                f"KeyError: method {mut} in project {project.full_name}.{module_data.module_i} which is tested by {etest_id} {test_e.cname}.{test_e.mname} is not found in source code."
                            )
                            filter_counter["no_mut"] += 1
                            continue
                        # etest
                        data = self.collect_sign_stmts(data, test_e)
                        data.test_method = test_e
                        data.test_e = test_e.raw_code
                        data.test_context = test_e.context
                        data.etype = test_e.exception
                        data.test_e_key = (
                            test_e.cname.replace(".", "/") + "#" + test_e.mname + "#()V"
                        )
                        # call stack
                        called_traces = []
                        called_traces: List[str] = extract_methods_traces_from_logs(
                            cov_d.mte
                        )
                        called_methods = map_traced_method_to_method_structure(
                            called_traces, p_methods_list, p_class_list, p_class_offset
                        )
                        data.call_stacks = called_methods

                        data.id = f"mut2e-{data_id}"
                        data_id += 1
                        proj_dataset.append(data)

                    save_dataset(data_dir, proj_dataset, append=True)
                    logger.info(
                        f"Size of the module {module_data.project}.{module_data.module} collected data is {len(proj_dataset)}"
                    )
            except KeyboardInterrupt:
                logger.warning(
                    f"Keyboard Interruption Error processing {module_data.module}"
                )
                proj_error_logs[
                    f"{module_data.project}.{module_data.module}"
                ] = traceback.format_exc()
                filter_counter["keyboard_interrupt"] += 1
                error += 1
            except su.TimeUtils.TimeoutException:
                logger.warning(f"Time out processing {module_data.module}")
                filter_counter["time_out"] += 1
                proj_error_logs[
                    f"{module_data.project}.{module_data.module}"
                ] = traceback.format_exc()
                error += 1
            except:
                error += 1
                proj_error_logs[
                    f"{module_data.project}.{module_data.module}"
                ] = traceback.format_exc()
                filter_counter["other_error"] += 1
            else:
                success += 1
            finally:
                pbar.update(1)
        logger.info(f"In total collected data {data_id} data.")
        pbar.close()
        su.io.rmdir(temp_dir)
        su.io.dump(
            data_dir / "filter_counter.json", filter_counter, su.io.Fmt.jsonPretty
        )
        su.io.dump(
            data_dir / "project_error.json", proj_error_logs, su.io.Fmt.jsonPretty
        )

    def extract_ne2e_data(
        self,
        results_dir: su.arg.RPath = Macros.results_dir / "coverage",
        data_dir: su.arg.RPath = Macros.work_dir / "data" / "ne2e",
        downloads_dir: su.arg.RPath = Macros.teco_downloads_dir,
    ):
        """
        Extract NE2E dataset
        """

        config = {
            k: v
            for k, v in locals().items()
            if k not in {"self", "results_dir", "data_dir", "downloads_dir"}
        }
        su.io.dump(data_dir / "config.json", config, su.io.Fmt.jsonPretty)

        Tool.ensure_tool_versions()
        Tool.require_compiled()

        raw_dataset: List[ModuleData] = su.io.load(results_dir / "data.pkl")
        temp_dir = su.io.mktmp_dir("etestgen")

        filter_counter = {
            "data_null": 0,
            "data_empty": 0,
            "no_mut": 0,
            "no_ex_method": 0,
            "max_mut_tok": 0,
            "max_em_tok": 0,
            "max_test_ne_tok": 0,
            "max_test_e_tok": 0,
            "min_mut_tok": 0,
            "min_test_ne_tok": 0,
            "min_test_e_tok": 0,
        }

        data_id = 0
        pbar = tqdm(total=len(raw_dataset))
        success = 0
        skip = 0
        error = 0
        for module_data in raw_dataset:
            pbar.set_description(f"total {data_id} (+{success} -{error} x{skip})")
            try:
                cov_data = module_data.mdata

                if (
                    len(cov_data.etest_tc_coverage) + len(cov_data.etest_unk_exception)
                    == 0
                ):
                    continue

                proj_dataset = []

                # make sure project is cloned and locate the module
                project = self.indexed_projects[module_data.project]
                project.clone(downloads_dir)
                project.checkout(project.data["sha"], forced=True)
                maven_project = MavenProject.from_project(project)
                maven_module = [
                    m for m in maven_project.modules if m.rel_path == module_data.module
                ][0]

                # collect all source code in the project
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
                method2toks = {}
                method2code = {}
                for record in records_methods:
                    method2toks[record["method"]] = record["tokens"]
                    method2code[record["method"]] = record["method_node"]
                # tokenize the tests and remove all assertions
                tests_to_process = [t.code for t in cov_data.tests]
                su.io.dump(temp_dir / "in.json", tests_to_process)
                su.io.dump(
                    temp_dir / "config.json",
                    {
                        "inPath": str(temp_dir / "in.json"),
                        "outPath": str(temp_dir / "out.json"),
                        "debugPath": str(temp_dir / "debug.txt"),
                    },
                )
                # Preparation
                p_method_data_dir = (
                    self.raw_data_dir / project.full_name / "joint.method.json"
                )
                p_class_data_dir = (
                    self.raw_data_dir / project.full_name / "joint.class.json"
                )
                p_methods_list = su.io.load(
                    p_method_data_dir, clz=List[MethodStructure]
                )
                p_class_list = su.io.load(p_class_data_dir, clz=List[ClassStructure])
                p_class_offset = p_class_list[0].id
                # setup
                etest_2_tc = {}
                for tc_id, eid_set in cov_data.tc_2_e.items():
                    for e_id in eid_set:
                        etest_2_tc[e_id] = tc_id
                test_2_cov = {
                    cov.test_i: cov
                    for cov in cov_data.etest_tc_coverage
                    + cov_data.etest_unk_exception
                    + cov_data.netest_coverage
                }
                mut_2_test_id = defaultdict(list)
                mut_2_etest_id = defaultdict(list)
                for test_id in cov_data.etest_cover_tc + cov_data.etest_unchecked:
                    # first build the mut 2 test map
                    cov_d = test_2_cov[test_id]
                    mut = self.extract_mut(cov_d.mte, [], method2code)
                    mut_2_etest_id[mut].append(test_id)
                #
                for test_id in cov_data.netest_cover_tc:
                    cov_d = test_2_cov[test_id]
                    mut = self.extract_mut(cov_d.mte, [], method2code)
                    mut_2_test_id[mut].append(test_id)
                #

                # find ne-e pairs
                ne_e_pairs = []  # (ne_id, e_id, mut)
                shared_muts = set(mut_2_etest_id.keys()) & set(mut_2_test_id.keys())
                for mut in shared_muts:
                    for ne_id in set(mut_2_test_id[mut]):
                        for e_id in set(mut_2_etest_id[mut]):
                            ne_e_pairs.append((ne_id, e_id, mut))
                #
                for ne_id, e_id, mut in ne_e_pairs:
                    data = DataNE2E()
                    # project info
                    data.project = project.full_name
                    data.module = module_data.module
                    data.module_i = module_data.module_i

                    # mut info
                    data.mut_key = mut
                    mut_constructor_key = mut.split("#")[0] + "#<init>#"
                    data.constructors = [
                        method2code[mt_key]
                        for mt_key in method2code
                        if mut_constructor_key in mt_key
                    ]
                    try:
                        data.mut_toks = method2toks[mut]
                        data.mut = method2code[mut]
                    except KeyError:
                        logger.warning(
                            f"KeyError: method {mut} in project {project.full_name}.{module_data.module_i} which is tested by etest {e_id} and netest {e_id} is not found in source code."
                        )
                        filter_counter["no_mut"] += 1
                        continue
                    # etest
                    test_e = cov_data.tests[e_id]
                    # data = self.collect_sign_stmts(data, test_e)
                    data.etest_sign = test_e.ast.get_sign()
                    data.etest_stmts = extract_stmts_from_ast(test_e.ast)
                    data.test_e = test_e.raw_code
                    data.etest_context = test_e.context
                    data.etype = test_e.exception
                    data.test_e_key = (
                        test_e.cname.replace(".", "/") + "#" + test_e.mname + "#()V"
                    )
                    # call stack
                    called_traces = []
                    called_traces: List[str] = extract_methods_traces_from_logs(
                        cov_d.mte
                    )
                    called_methods = map_traced_method_to_method_structure(
                        called_traces, p_methods_list, p_class_list, p_class_offset
                    )
                    data.call_stacks = called_methods

                    # netest
                    test_ne = cov_data.tests[ne_id]
                    data.netest_sign = test_ne.ast.get_sign()
                    data.netest_stmts = extract_stmts_from_ast(test_ne.ast)
                    data.test_ne = test_ne.raw_code
                    data.ntest_context = test_ne.context
                    data.test_ne_key = (
                        test_ne.cname.replace(".", "/") + "#" + test_ne.mname + "#()V"
                    )

                    data.id = f"ne2e-{data_id}"
                    data_id += 1
                    proj_dataset.append(data)

                save_dataset(data_dir, proj_dataset, append=True)
                logger.info(
                    f"Size of the module {module_data.project}.{module_data.module} collected data is {len(proj_dataset)}"
                )
            except KeyboardInterrupt:
                raise
            except:
                error += 1
                logger.warning(
                    f"Error processing {module_data.module}: {traceback.format_exc()}"
                )
                filter_counter["data_null"] += 1
            else:
                success += 1
            finally:
                pbar.update(1)
        pbar.close()

        su.io.rmdir(temp_dir)
        su.io.dump(
            data_dir / "filter_counter.json", filter_counter, su.io.Fmt.jsonPretty
        )


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.INFO)
    CLI(ThrowsCoverageAnalyzer, as_positional=False)
