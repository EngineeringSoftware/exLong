import logging.config
import os
import re
import shutil
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import seutil as su
import textwrap
from jsonargparse import CLI
from seutil.maven import MavenModule
from seutil.project import Project

from etestgen.codellama import CodeLLaMA, DataProcessor, QuantLLaMA
from etestgen.data.data import DataNE2E
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.eval import (
    collect_stack_trace_from_netest,
    compute_etest_coverage,
    compute_test_coverage,
    extract_etest_data_from_coverage,
    get_ne_data,
)
from etestgen.eval.extract_etest_data_from_coverage import (
    collect_module_source_code,
)
from etestgen.llm import eval as llm_eval
from etestgen.llm.utils import LLMResults
from etestgen.macros import Macros

logger = su.log.get_logger(__name__, su.log.WARNING)
RUN_METRIC_ORDER = ["coverage", "runnable-overall", "compilable"]


class ExlongTimoutException(Exception):
    """Exception raised when exlong timeout is reached"""


def timeout_handler(signum, frame):
    """throw exception when timeout is reached"""
    raise ExlongTimoutException("Timeout reached")


class ExlongCli:
    def __init__(
        self,
        repos_file_name: str = "repos.json",
        output_path: str | None = None,
    ):
        self._repos_file_path = (
            Macros.work_dir / "repos" / "filtered" / repos_file_name
        )
        self._config_file_path = Macros.config_dir / "cli-no-name.yaml"
        self._config: Dict[str, Any] = su.io.load(self._config_file_path)  # type: ignore
        self._prediction_dir = (
            Macros.exp_dir
            / self._config["setup"]
            / self._config["model_name"]
            / "rq2-results"
        )
        self._output_path = output_path
        self._use_quant = True

    def setup_model(self, model_type: str):
        QuantLLaMA.QuantLlama.setup_model(model_type=model_type)

    def generate_project_info(
        self,
        repo_path: Optional[str] = None,
        repo_link: Optional[str] = None,
        sha: str = "",
    ):
        if repo_path is not None:
            out_proj = Project(
                full_name=Path(repo_path).name,
                url=f"file://{repo_path}",
            )
        elif repo_link is not None:
            out_proj = Project.from_github_url(repo_link)  # type: ignore
        else:
            raise ValueError("must provide repo path or repo link")

        out_proj.clone(Macros.work_dir / "coverage-new")
        out_proj.data["sha"] = sha
        su.io.dump(
            self._repos_file_path,
            [out_proj],
        )

    def generate_coverage_data(self):
        project = su.io.load(self._repos_file_path, clz=List[Project])[0]
        etest_cov_collector = (
            compute_etest_coverage.EtestCoverageComputer()
        )
        test_cov_collector = compute_test_coverage.TestCoverageCollector()
        stacktrace_collector = (
            collect_stack_trace_from_netest.TestStackTraceCollector()
        )
        etest_cov_collector.compute_coverage_data(
            repos_file=self._repos_file_path
        )
        test_cov_collector.compute_netest_coverage(
            repos_file=self._repos_file_path
        )
        stacktrace_collector.compute_netest_coverage(
            out_dir=Macros.work_dir / "coverage-new",
            repos_file=self._repos_file_path,
        )
        stacktrace_collector.collect_project_stack_trace_with_etypes(
            project,
            out_dir=Macros.work_dir / "coverage-new" / project.full_name,
        )

    def generate_machine_view_dataset(self):
        if os.path.isdir(Macros.data_dir / "new-rq2-public"):
            shutil.rmtree(Macros.data_dir / "new-rq2-public")
        if os.path.isdir(Macros.data_dir / "new-rq2-public-new"):
            shutil.rmtree(Macros.data_dir / "new-rq2-public-new")
        nedata_collector = get_ne_data.NEData()
        nedata_collector.get_pub_with_throw(self._repos_file_path)
        nedata_collector.get_non_cover_ne2e_data()

        coverage_analyzer = extract_etest_data_from_coverage.CoverageAnalyzer()
        coverage_analyzer.extract_conditions(dataset="new-rq2-public")

        shutil.move(
            Macros.data_dir / "new-rq2-public",
            Macros.data_dir / "new-rq2-public-new",
        )

    def generate_user_view_data(
        self,
        mut_file_path: str,
        mut_line: int,
        file_path: str,  # only need the outer most class name
        line: int,
        test_context_path: str,
        test_name: Optional[str] = None,
    ):
        if os.path.isdir(Macros.data_dir / "new-rq2-public"):
            shutil.rmtree(Macros.data_dir / "new-rq2-public")
        if os.path.isdir(Macros.data_dir / "new-rq2-public-new"):
            shutil.rmtree(Macros.data_dir / "new-rq2-public-new")
        file_full_path = str(Path(file_path).resolve())
        class_root = file_full_path.find("src/main/java/") + len(
            "src/main/java/"
        )
        class_name = file_full_path[class_root:-5].replace("/", ".")

        mut_file_full_path = str(Path(mut_file_path).resolve())
        mut_class_root = mut_file_full_path.find("src/main/java/") + len(
            "src/main/java/"
        )
        mut_class_name = mut_file_full_path[mut_class_root:-5].replace(
            "/", "."
        )
        target_proj: Project = su.io.load(
            self._repos_file_path,
            clz=List[Project],  # type: ignore
        )[0]
        with open(test_context_path) as test_context_file:
            test_context = test_context_file.read()

        project_dir = Macros.work_dir / "coverage-new" / target_proj.full_name
        out: List[DataNE2E] = []

        for module_dir in project_dir.iterdir():
            if not module_dir.is_dir():
                continue
            maven_module: MavenModule = su.io.load(
                module_dir / "module.yaml",
                clz=MavenModule,
            )  # type: ignore
            maven_module.project = target_proj  # type: ignore
            module_i = int(os.path.basename(module_dir))

            try:
                method_records: List[Dict[str, Any]] = su.io.load(
                    module_dir / "method_records.jsonl"
                )  # type: ignore
            except FileNotFoundError:
                method_records: List[Dict[str, Any]] = (
                    collect_module_source_code(
                        target_proj,
                        Macros.work_dir / "downloads",
                        maven_module.rel_path,
                    )
                )
                su.io.dump(module_dir / "method_records.jsonl", method_records)
            method = None
            for m in method_records:
                if m["fqCName"].split("$")[0].split("#")[0] == mut_class_name:
                    if m["startLine"] == mut_line:
                        method = m
            if method is None:
                raise ValueError(f"No not found in {mut_file_path}:{mut_line}")
            for m in method_records:
                if m["fqCName"].split("$")[0].split("#")[0] == class_name:
                    for ex in m["exceptions"]:
                        if int(ex.split("@@")[1]) == line:
                            dt = DataNE2E()
                            dt.id = 0  # type: ignore
                            dt.project = target_proj.full_name
                            dt.module_i = module_i
                            dt.module = maven_module.rel_path
                            dt.mut_key = method["method"]
                            dt.mut = method["method_node"]
                            ex_line = int(ex.split("@@")[1]) - int(
                                m["startLine"]
                            )
                            dt.e_stack_trace = [[m, ex_line]]  # type: ignore
                            dt.etype = ex.split("@@")[0]
                            dt.test_context = test_context
                            if test_name is not None:
                                dt.test_e_key = (
                                    "placeholder#" + test_name + "#()V"
                                )
                            out.append(dt)
                            break

        if len(out) == 0:
            raise ValueError(
                f"No exception found at {class_name} line: {line}"
            )
        save_dataset(Macros.data_dir / "new-rq2-public", out, clz=DataNE2E)

        self.add_real_stack_trace()
        nedata_collector = get_ne_data.NEData()
        nedata_collector.collect_netest_data(
            mut2e_data_dir=Macros.data_dir / "new-rq2-public-new"
        )
        coverage_analyzer = extract_etest_data_from_coverage.CoverageAnalyzer()
        coverage_analyzer.extract_conditions(dataset="new-rq2-public-new")

    def add_real_stack_trace(self):
        stack_collector = (
            collect_stack_trace_from_netest.TestStackTraceCollector()
        )
        stack_collector.add_real_stack_traces_to_eval_data(
            out_dir=Macros.work_dir / "coverage-new",
            repos_file=self._repos_file_path,
            data_dir="new-rq2-public",
        )
        # shutil.move(Macros.data_dir / "new-rq2-public",
        #             Macros.data_dir / "new-rq2-public-old")
        # shutil.move(Macros.data_dir / "new-rq2-public-new",
        #             Macros.data_dir / "new-rq2-public")

    def generate_prompt(self):
        su.io.rm(Macros.setup_dir / "cli-tool" / "eval" / "test")
        shutil.copytree(
            Macros.data_dir / "new-rq2-public-new",
            Macros.setup_dir / "cli-tool" / "eval" / "test",
        )
        data_processor = DataProcessor.DataProcessor(
            config_file=self._config_file_path
        )
        data_processor.process_test_data()

    def run_inference(self):
        if self._use_quant:
            codellama = QuantLLaMA.QuantLlama(
                config_file=self._config_file_path
            )
        else:
            codellama = CodeLLaMA.CodeLlama(config_file=self._config_file_path)
        codellama.run_gen()

    def run_eval(self):
        evaluator = llm_eval.LLMCodeEvaluator(
            config_file=self._config_file_path
        )
        evaluator.eval_runtime_metrics()

        # replace topk with best sample
        llm_outputs = load_dataset(
            save_dir=self._prediction_dir,
            clz=LLMResults,
        )
        for res in llm_outputs:
            # select best test if non is > 0
            # then go to next metric
            selection_metric = RUN_METRIC_ORDER[0]
            for metric in RUN_METRIC_ORDER:
                if res.metrics[f"{metric}-max"] > 0:
                    selection_metric = metric
                    break
            seleted_sample_id = np.argmax(
                res.metrics[f"{selection_metric}-topk"]
            )
            print(seleted_sample_id)
            res.topk = [res.topk[seleted_sample_id]]

        save_dataset(self._prediction_dir, llm_outputs)

    def add_to_file(self):
        assert self._output_path is not None
        llm_outputs = load_dataset(
            save_dir=self._prediction_dir,
            clz=LLMResults,
        )

        with open(self._output_path, "w") as out_file:
            for res in llm_outputs:
                out_file.write(res.topk[0])
                out_file.write("\n\n")

    def add_to_test_suit(self, test_context_path: str):
        with open(test_context_path) as test_file:
            test_context = test_file.read()
        class_name = Path(test_context_path).stem
        class_match = re.search(
            r"class\s+" + class_name + r".*?\{\n*([\S\s]*)\n*\}", test_context
        )
        assert class_match is not None
        indents = re.findall(r"^([ \t]*).*", class_match.group(1), flags=re.MULTILINE)
        indents = []
        for line in class_match.group(1).splitlines():
            if len(line) != 0:
                indent_match = re.match(r"([ \t]*).*", line)
                if indent_match is not None:
                    indents.append(indent_match.group(1))
        indent_level = min(indents, key=len)


        llm_outputs = load_dataset(
            save_dir=self._prediction_dir,
            clz=LLMResults,
        )
        methods = textwrap.dedent("""
        /***
        ------------ Test genereated by exLong below ------------
        ***/

        """)
        methods += "\n\n".join([res.topk[0] for res in llm_outputs])
        methods += textwrap.dedent("""

        /***
        ------------ Test genereated by exLong above ------------
        ***/
        """)
        methods = "\n" + textwrap.indent(methods, indent_level) + "\n"
        new_class = class_match.group(0)[:-1] + methods + "}"
        new_test = test_context[:class_match.start()] + new_class + test_context[class_match.end():]
        with open(test_context_path, "w") as test_file:
            test_file.write(new_test)

    def machine_view(
        self,
        repo_path: Optional[str] = None,
        repo_link: Optional[str] = None,
        sha: str = "",
        pick_best: bool = False,
        output_path: Optional[str] = None,
        timeout: Optional[int] = None,
        quant: Optional[bool] = None,
        test_context_path: str | None = None,
    ):
        if test_context_path is None and output_path is None:
            raise ValueError("test_context_path and output_path cannot both be none")
        if timeout is not None:
            signal.signal(signal.SIGALRM, handler=timeout_handler)
            signal.alarm(timeout)
        try:
            if output_path is not None:
                self._output_path = output_path
            self.generate_project_info(
                repo_path=repo_path, repo_link=repo_link, sha=sha
            )
            self.generate_coverage_data()
            self.generate_machine_view_dataset()
            self.generate_prompt()
            if quant is not None:
                self._use_quant = quant
            self.run_inference()
            if pick_best:
                self.run_eval()
            if output_path is not None:
                self.add_to_file()
            elif test_context_path is not None:
                self.add_to_test_suit(test_context_path)

        except ExlongTimoutException:
            logger.error("Time out reached")

    def user_view(
        self,
        mut_file_path: str,
        mut_line: int,
        throw_file_path: str,
        throw_line: int,
        test_context_path: str,
        repo_path: Optional[str] = None,
        repo_link: Optional[str] = None,
        test_name: Optional[str] = None,
        sha: str = "",
        pick_best: bool = False,
        output_path=None,
        quant: Optional[bool] = None,
        regenerate_data: bool = True,
    ):
        if output_path is not None:
            self._output_path = output_path
        if test_name is not None:
            # if name is given switch to with name mode
            self._config_file_path = Macros.config_dir / "cli-with-name.yaml"

        self.generate_project_info(
            repo_path=repo_path,
            repo_link=repo_link,
            sha=sha,
        )

        if regenerate_data:
            self.generate_coverage_data()
            self.generate_user_view_data(
                mut_file_path,
                mut_line,
                throw_file_path,
                throw_line,
                test_context_path,
                test_name=test_name,
            )

        if pick_best:
            self._config["data-type"] += "-ALL"
        self.generate_prompt()
        if quant is not None:
            self._use_quant = quant
        self.run_inference()
        if pick_best:
            self.run_eval()
        if output_path is not None:
            self.add_to_file()
        elif test_context_path is not None:
            self.add_to_test_suit(test_context_path)


if __name__ == "__main__":
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
        }
    )
    CLI(ExlongCli, as_positional=False)
