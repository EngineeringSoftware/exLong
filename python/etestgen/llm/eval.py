import traceback
from seutil.maven import MavenModule, MavenProject
from typing import *
from seutil.project import Project
from jsonargparse import CLI
from pathlib import Path
import seutil as su
import os

# os.environ["SP_DIR"] = "/opt/anaconda3/envs/etestgen/lib/python3.8/site-packages"
import subprocess
from tqdm import tqdm
import collections

from etestgen.eval.compute_throws_coverage import TestMethod, scan_manual_tests
from etestgen.macros import Macros
from etestgen.data.utils import load_dataset, tokenize_code, save_dataset
from etestgen.llm.utils import (
    LLMResults,
    extract_code_from_response,
    extract_test_from_raw_code,
)
from etestgen.data.structures import AST
from etestgen.data.data import DataMUT2E, DataNE2E
from etestgen.data.tool import Tool
from etestgen.utils import (
    aggregate_metrics,
    summarize_metrics,
    compute_similarity_metrics,
    summarize_topk_metrics,
)
from etestgen.eval.compute_etest_coverage import add_try_catch_block

logger = su.log.get_logger(__name__)


class LLMCodeEvaluator:
    DOWNLOADS_DIR = Macros.downloads_dir
    test_placeholder = "/*TEST PLACEHOLDER*/"

    def __init__(
        self,
        config_file: Union[str, Path],
        eval_set: str = "test",
        train_seed: Optional[int] = None,
    ):
        self.config = su.io.load(config_file)
        self.setup = self.config["setup"]
        self.exp = self.config["model_name"]
        if train_seed is not None:
            self.exp += f"-{train_seed}"
        self.prediction_dir = (
            Macros.exp_dir / self.setup / self.exp / f"{eval_set}-results"
        )
        self.eval_out_dir = Macros.exp_dir / self.setup / self.exp / f"{eval_set}-out"
        repos_file = Macros.work_dir / "repos" / "filtered" / "repos.json"
        self.model_results_dir = Macros.results_dir / "model-results"
        su.io.mkdir(self.eval_out_dir)
        eval_set = self.config.get("split", "test")
        self.eval_set = f"eval/{eval_set}"
        if eval_set == "real-test":
            self.eval_set = "real-eval/test"
        elif eval_set == "subset":
            self.eval_set = "subset/test"
        data_dir = Macros.work_dir / self.config["test_data"]
        with tqdm(desc="Loading dataset") as pbar:
            self.dataset = load_dataset(data_dir, clz=DataNE2E, pbar=pbar)
        self.projects: List[Project] = su.io.load(repos_file, clz=List[Project])

    def run_evaluation(self, which: str):
        """
        Entry point for the LLM evaluation.
        """
        if which == "runtime-metrics":
            self.eval_runtime_metrics()
        elif which == "sim-metrics":
            self.eval_llm_sim()
        else:
            raise RuntimeError

    def eval_llm_sim(self, selected_ids: List[Any] = None):
        """
        Evaluate LLM predictions w.r.t. the similarity.
        """

        llm_preds: List[LLMResults] = self._load_preds()
        pred_metrics = []
        model_preds = []
        total_examples = (
            min(len(llm_preds), len(selected_ids)) if selected_ids else len(llm_preds)
        )
        with tqdm(total=total_examples, desc="evaluating similarity metrics") as pbar:
            for llm_result in llm_preds:
                if selected_ids and llm_result.id not in selected_ids:
                    continue
                topk_methods_toks = []
                target_test_name = llm_result.input.test_e_key.split("#")[1]
                # compute the similarity metrics for topk predictions
                new_topk_preds = []
                for raw_pred_method in llm_result.topk:
                    # extract code
                    llm_written_code = extract_code_from_response(
                        response=raw_pred_method
                    )
                    if self.config["model_name"] == "catlm":
                        predicted_test = llm_written_code
                    else:
                        predicted_test = extract_test_from_raw_code(
                            llm_written_code, target_test_name
                        )
                    raw_pred_method = predicted_test.strip()
                    topk_methods_toks.append(raw_pred_method.split())
                    new_topk_preds.append(raw_pred_method)
                llm_result.topk = new_topk_preds
                gold_method_toks = llm_result.input.test_e.split()
                # compute similarity metrics
                similarity_metrics = compute_similarity_metrics(
                    gold=gold_method_toks, topk=topk_methods_toks
                )
                model_preds.append(raw_pred_method)
                pred_metrics.append(similarity_metrics)
                pbar.update(1)

        metrics_summary = summarize_metrics(aggregate_metrics(pred_metrics))
        su.io.dump(
            self.eval_out_dir / "similarity_metrics_summary.json",
            metrics_summary,
            su.io.Fmt.jsonPretty,
        )
        su.io.dump(
            self.eval_out_dir / "model_preds.jsonl",
            model_preds,
        )
        if selected_ids:
            results_file_name = f"selected-{len(selected_ids)}-{self.setup}-{self.exp}-{self.eval_set}-sim-metrics.json".replace(
                "/", "-"
            )
        else:
            results_file_name = (
                f"{self.setup}-{self.exp}-{self.eval_set}-sim-metrics.json".replace(
                    "/", "-"
                )
            )
        su.io.dump(
            self.model_results_dir / results_file_name,
            metrics_summary,
            su.io.Fmt.jsonPretty,
        )
        save_dataset(self.prediction_dir, llm_preds, clz=LLMResults)

    def eval_runtime_metrics(self, selected_ids: List[Any] = None) -> List[dict]:
        """
        Compute and summarize the runtime metrics for LLM generated tests.
        """

        llm_results: List[LLMResults] = self._load_preds()
        if selected_ids:
            llm_results = [res for res in llm_results if res.id in selected_ids]
        # build prj to modules
        prj_to_modules = collections.defaultdict(set)
        for llm_res in llm_results:
            prj_to_modules[llm_res.project].add(int(llm_res.module_i))
        eval_prj_set = list(prj_to_modules.keys())
        projects_for_evaluation = [
            p for p in self.projects if p.full_name in eval_prj_set
        ]
        run_time_metrics = []
        llm_results_with_metrics = []
        with tqdm(
            total=len(projects_for_evaluation),
            desc=f"In total {len(projects_for_evaluation)} projects to evaluate on.",
        ) as pbar:
            for prj in projects_for_evaluation:
                # target modules
                target_modules = prj_to_modules[prj.full_name]
                # set up environment
                prj.clone(self.DOWNLOADS_DIR)
                with su.io.cd(prj.dir):
                    # su.bash.run(f"rm -rf {prj.dir}/.git/index.lock")
                    su.bash.run("git clean -ffdx")
                    prj.checkout(prj.data["sha"], forced=True)
                maven_proj = MavenProject.from_project(prj)
                for module_i, _ in enumerate(maven_proj.modules):
                    if module_i not in target_modules:
                        continue
                    (
                        module_results,
                        llm_results_new,
                    ) = self.evaluate_module_tests_on_runtime_metrics(
                        project=prj, module_i=module_i, all_predictions=llm_results
                    )
                    run_time_metrics.extend(module_results)
                    llm_results_with_metrics.extend(llm_results_new)
                pbar.update(1)
        runtime_metrics_summary = summarize_metrics(aggregate_metrics(run_time_metrics))
        llm_results_with_metrics = sorted(llm_results_with_metrics, key=lambda x: x.id)
        save_dataset(self.prediction_dir, llm_results_with_metrics, clz=LLMResults)
        if selected_ids:
            results_file_name = f"selected-{len(selected_ids)}-{self.setup}-{self.exp}-{self.eval_set}-runtime-metrics.json".replace(
                "/", "-"
            )
        else:
            results_file_name = (
                f"{self.setup}-{self.exp}-{self.eval_set}-runtime-metrics.json".replace(
                    "/", "-"
                )
            )
        runtime_metrics_summary["runnable-overall"] = runtime_metrics_summary[
            "runnable-overall-max"
        ]
        su.io.dump(
            self.model_results_dir / results_file_name,
            runtime_metrics_summary,
            su.io.Fmt.jsonPretty,
        )
        return run_time_metrics

    def eval_subset_llm_results(
        self, subset_id_file: Union[str, Path], metric_type: str = "runtime"
    ):
        """
        Extract the subset results for reporting the performance on the interseced projects.
        """
        subset_ids: List[int] = su.io.load(subset_id_file)
        llm_results_with_metrics = load_dataset(
            self.prediction_dir, clz=LLMResults, expected_ids=set(subset_ids)
        )
        metrics = []
        for llm_res in llm_results_with_metrics:
            metrics.append(llm_res.metrics)
        metrics_summary = summarize_metrics(aggregate_metrics(metrics))
        if metric_type == "runtime":
            metrics_summary["runnable-overall"] = metrics_summary[
                "runnable-overall-max"
            ]
        results_file_name = f"{self.setup}-{self.exp}-{self.eval_set}-intersect-{metric_type}-metrics.json".replace(
            "/", "-"
        )
        su.io.dump(
            self.model_results_dir / results_file_name,
            metrics_summary,
            su.io.Fmt.jsonPretty,
        )

    def eval_code_llama_sim(self, pred_file: Path):
        """
        Compute the similarity metrics between codellama prediction and gold test.
        """

        llama_preds = su.io.load(pred_file)
        llm_preds: List[LLMResults] = self._load_preds()
        model_predicted_tests = []
        pred_metrics = []

        with tqdm(total=len(llm_preds), desc="evaluating similarity metrics") as pbar:
            for llm_result, llama_pred in zip(llm_preds, llama_preds):
                target_test_name = llm_result.mname
                # extract code
                llm_written_code = extract_code_from_response(response=llama_pred)
                predicted_test = extract_test_from_raw_code(
                    llm_written_code, target_test_name
                )
                raw_pred_method = predicted_test.strip()
                gold_method_toks = llm_result.input.test_e.split()
                # compute similarity metrics
                similarity_metrics = compute_similarity_metrics(
                    gold=gold_method_toks, topk=[raw_pred_method.split()]
                )
                model_predicted_tests.append(raw_pred_method)
                pred_metrics.append(similarity_metrics)
                pbar.update(1)
        metrics_summary = summarize_metrics(aggregate_metrics(pred_metrics))
        su.io.dump(
            self.eval_out_dir / "similarity_metrics_summary.json",
            metrics_summary,
            su.io.Fmt.jsonPretty,
        )
        su.io.dump(
            self.eval_out_dir / "model_preds.jsonl",
            model_predicted_tests,
        )

    ########################
    # General helpers #
    ########################
    def _load_preds(self) -> List[LLMResults]:
        """
        Load the LLM's predictions from disk to memory
        """

        llm_outputs = load_dataset(save_dir=self.prediction_dir, clz=LLMResults)
        return llm_outputs

    ########################
    # Compute runtime metrics helpers #
    ########################

    def extract_manual_tests(
        self, maven_module: MavenModule, work_dir: Path, out_dir: Path
    ):
        test_config = {
            "classpath": maven_module.exec_classpath,
            "mainSrcRoot": maven_module.main_srcpath,
            "testSrcRoot": maven_module.test_srcpath,
            "outPath": str(work_dir / "manual.out.json"),
            "debugPath": str(out_dir / "debug.txt"),
        }
        test_config_path = work_dir / "manual.config.json"
        su.io.dump(test_config_path, test_config)
        try:
            su.bash.run(
                f"java -cp {Tool.core_jar} org.etestgen.core.SrcTestScanner {test_config_path}",
                0,
            )
        except:
            logger.info("Failed to extract manual tests for %s", maven_module)
            breakpoint()
            return []
        tests = su.io.load(work_dir / "manual.out.json", clz=List[TestMethod])
        su.io.dump(out_dir / "manual.tests.jsonl", tests)
        return tests

    def add_throws_keyword(
        self,
        test_file_content: str,
    ):
        """
        Add 'throws Exception' to the generated test method.
        """
        temp_dir = su.io.mktmp_dir("etestgen")
        su.io.dump(temp_dir / "temp.java", test_file_content, su.io.Fmt.txt)
        test_config = {
            "inFile": str(temp_dir / "temp.java"),
            "outPath": str(temp_dir / "temp.java"),
        }
        test_config_path = temp_dir / "manual.config.json"
        su.io.dump(test_config_path, test_config)
        try:
            su.bash.run(
                f"java -cp {Tool.core_jar} org.etestgen.core.AddThrowModifier {test_config_path}",
                0,
            )
        except:
            # logger.warning("Failed to add throws keyword to %s", test_file_content)
            pass

        test = su.io.load(temp_dir / "temp.java", su.io.Fmt.txt)
        su.io.rmdir(temp_dir)
        return test

    def evaluate_module_tests_on_runtime_metrics(
        self, project: Project, module_i: int, all_predictions: List[LLMResults]
    ) -> List[dict]:
        """
        Compute the runtime metrics for the generated tests for the given module.
        """

        runtime_out_dir = self.eval_out_dir / "runtime_logs"
        out_dir = runtime_out_dir / project.full_name
        su.io.mkdir(out_dir)

        predicted_tests: List[LLMResults] = self.extract_predicted_tests_for_module(
            predictions=all_predictions, project=project, module_i=module_i
        )
        if len(predicted_tests) == 0:
            logger.warning("No predictions for %s-%d", project.full_name, module_i)
            return []
        ground_truths = [llm_output.input.test_method for llm_output in predicted_tests]

        # prepare maven project to run the tests
        maven_proj = self.prepare_maven_project(project)
        su.io.dump(out_dir / "maven.yaml", maven_proj)
        maven_module = maven_proj.modules[module_i]

        # prepare work dir and out dir
        work_dir = su.io.mktmp_dir("etestgen")
        module_out_dir = out_dir / f"{module_i}"
        su.io.dump(module_out_dir / "module.yaml", maven_module)
        # recompile project
        self.checkout_and_compile_project(maven_proj, project)
        # compute runtime metrics
        (
            module_runtime_metrics_summary,
            llm_results,
        ) = self.compute_runtime_metrics_for_module(
            maven_module, work_dir, module_out_dir, predicted_tests, ground_truths
        )
        su.io.rmdir(work_dir)
        return module_runtime_metrics_summary, llm_results

    def compute_runtime_metrics_for_module(
        self,
        maven_module: MavenModule,
        work_dir: Path,
        out_dir: Path,
        llm_outputs: List[LLMResults],
        ground_truths: List[TestMethod],
    ):
        # run tests and get the metrics
        module_runtime_metrics = []
        for gold_test, llm_output in zip(ground_truths, llm_outputs):
            metrics = self.compute_topk_runtime_metrics(
                llm_output, maven_module, out_dir, work_dir, gold_test
            )
            llm_output.metrics = metrics
            module_runtime_metrics.append(metrics)
        #
        return module_runtime_metrics, llm_outputs

    def extract_predicted_tests_for_module(
        self, predictions: List[LLMResults], project: Project, module_i: int
    ) -> List[LLMResults]:
        """
        Extract the LLM generated tests for the module of the given project.
        """

        predicted_tests: List[LLMResults] = [
            pred
            for pred in predictions
            if pred.project == project.full_name
            and pred.module_i == module_i
            and pred.topk != []
        ]
        return predicted_tests

    def prepare_maven_project(self, project: Project):
        """
        Prepare maven project.
        """

        maven_proj = MavenProject.from_project(project)
        maven_proj.backup_pom()
        maven_proj.hack_pom_delete_plugin("maven-checkstyle-plugin")
        maven_proj.compile()
        maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")

        return maven_proj

    def checkout_and_compile_project(self, maven_proj: MavenProject, project: Project):
        """
        Checkout and compile the project.
        """

        maven_proj.restore_pom()
        project.checkout(project.data["sha"], forced=True)
        with su.io.cd(project.dir):
            su.bash.run("git clean -ffdx")
        maven_proj.compile()
        maven_proj.backup_pom()
        maven_proj.hack_pom_delete_plugin("maven-dependency-plugin")

    def compute_topk_runtime_metrics(
        self,
        llm_output: LLMResults,
        maven_module: MavenModule,
        out_dir: Path,
        work_dir: Path,
        gold_test: TestMethod,
    ) -> dict:
        """
        Compute and summarize the runtime metrics for the topk predictions from LLMs.
        """

        topk_metrics = collections.defaultdict(list)
        for k, predicted_test in enumerate(llm_output.topk):
            # compute runtime metrics
            runtime_metrics = self.compute_test_runtime_metrics(
                maven_module=maven_module,
                work_dir=work_dir,
                out_dir=out_dir,
                predicted_test=predicted_test,
                target_exception=llm_output.input.etype,
                gold_test=gold_test,
                ci=k,
                stack_trace=llm_output.input.e_stack_trace,
            )

            for k, v in runtime_metrics.items():
                topk_metrics[k].append(v)
        # aggregate topk metrics
        sum_metrics = summarize_topk_metrics(topk_metrics)
        return sum_metrics

    def compute_test_runtime_metrics(
        self,
        maven_module: MavenModule,
        work_dir: Path,
        out_dir: Path,
        predicted_test: Tuple[str, str],
        gold_test: TestMethod,
        target_exception: str,
        ci: int = 0,
        timeout_limit: int = 10,
        stack_trace: Any = None,
    ) -> dict:
        """
        Run the LLM-generated tests and report metrics (compilable, runnable, timeout).
        """

        (
            compilable,
            runnable,
            timeout,
            coverage,
        ) = (
            1,
            1,
            0,
            0,
        )
        metrics = {}
        classpath = os.pathsep.join(
            [
                maven_module.main_classpath,
                maven_module.test_classpath,
                maven_module.dependency_classpath,
            ]
        )

        # extract the test to an ad-hoc test class
        run_path = work_dir / "run"
        su.io.mkdir(run_path, fresh=True)
        if gold_test is None:
            package = None
            test_name = "adhoc_Test"
            test_path = run_path / f"{test_name}.java"
            ccontext = (
                "import static org.junit.Assert.*;\nimport org.junit.Test;\n\npublic class Test {\n"
                + self.test_placeholder
                + " }"
            )
        else:
            package = ".".join(gold_test.cname.split(".")[0:-1])
            csname = "adhoc_" + gold_test.cname.split(".")[-1]
            test_name = package + "." + csname
            test_path = run_path / package.replace(".", "/") / f"{csname}.java"
            ccontext = gold_test.ccontext
        test_file_content = ccontext.replace(self.test_placeholder, predicted_test)
        Tool.ensure_tool_versions()
        Tool.require_compiled()
        new_test_file_content = self.add_throws_keyword(test_file_content)
        su.io.dump(test_path, new_test_file_content, su.io.Fmt.txt)
        # compile and run the test
        with su.io.cd(run_path):
            # compile the test
            rr = su.bash.run(f"javac -cp {classpath} {test_path}")

            if rr.returncode != 0:
                su.io.dump(
                    out_dir / f"eval.error" / f"test-{ci}.java",
                    new_test_file_content,
                    su.io.Fmt.txt,
                )
                su.io.dump(
                    out_dir / f"eval.error" / f"test-{ci}.log",
                    "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                    su.io.Fmt.txt,
                )

                metrics.update({"compilable": 0, "runnable-overall": 0, "coverage": 0})
                return metrics

            # run the test, if it does not include the expected exception, ignore it
            compilable = 1

            # scan test
            maven_module.test_srcpath = str(run_path)
            temp_dir = su.io.mktmp_dir("etestgen")
            test_methods = self.extract_manual_tests(maven_module, temp_dir, temp_dir)
            su.io.rm(temp_dir)
            if len(test_methods) > 1:
                logger.warning(
                    f"{len(test_methods)} exception test methods are extracted"
                )
            elif len(test_methods) == 0:
                logger.warning("No test method is extracted")
                metrics.update(
                    {"compilable": 1, "match": 0, "runnable-overall": 0, "coverage": 0}
                )
                return metrics
            if (
                target_exception != test_methods[0].exception
                and target_exception.split(".")[-1] != test_methods[0].exception
            ):
                runnable = 0
                metrics.update(
                    {
                        "compilable": 1,
                        "match": 0,
                        "runnable-overall": 0,
                        "coverage": 0,
                    }
                )
                return metrics
            else:
                try:
                    rr = su.bash.run(
                        f"java -cp .:{Tool.rt_jar}:{classpath} -ea org.junit.runner.JUnitCore {test_name}",
                        timeout=timeout_limit,
                    )
                except subprocess.TimeoutExpired:
                    su.io.dump(
                        out_dir / f"eval.error" / f"test-{ci}.java",
                        new_test_file_content,
                        su.io.Fmt.txt,
                    )
                    su.io.dump(
                        out_dir / f"eval.error" / f"test-{ci}.log",
                        "TIMEOUT",
                        su.io.Fmt.txt,
                    )
                    runnable = 0
                    timeout = 1
                    coverage = 0
                except:
                    rr.returncode = 1
                if rr.returncode != 0:
                    su.io.dump(
                        out_dir / f"eval.error" / f"test-{ci}.java",
                        new_test_file_content,
                        su.io.Fmt.txt,
                    )
                    su.io.dump(
                        out_dir / f"eval.error" / f"test-{ci}.log",
                        "STDOUT\n" + rr.stdout + "\n\nSTDERR\n" + rr.stderr,
                        su.io.Fmt.txt,
                    )
                    runnable = 0
                    coverage = 0
                if rr.returncode == 0 and timeout != 1:
                    # check if throw statement is covered
                    coverage = self.collect_throw_coverage(
                        test_methods[0],
                        work_dir,
                        ccontext,
                        classpath,
                        test_path,
                        stack_trace,
                    )

            metrics.update(
                {
                    "compilable": compilable,
                    "runnable": runnable,
                    "timeout": timeout,
                    "match": 1,
                    "coverage": coverage,
                    "runnable-overall": runnable,
                }
            )
        return metrics

    def collect_throw_coverage(
        self,
        test_method: TestMethod,
        work_dir: Path,
        ccontext: str,
        classpath: str,
        test_path: str,
        stack_trace: List,
    ) -> int:
        """
        Run the generated test to see if it covers the target throw statement as user.
        """
        modified_test, test_file_content = add_try_catch_block(
            test_method.code,
            ccontext,
            work_dir / "stack-trace-logs.txt",
        )
        test_file_content = test_file_content.replace(
            self.test_placeholder, modified_test
        )
        su.io.dump(test_path, test_file_content, su.io.Fmt.txt)
        # run
        run_path = work_dir / "run"
        ex_log = work_dir / "stack-trace-logs.txt"
        su.io.rm(ex_log)
        coverage = 0
        try:
            # compile and run the test
            with su.io.cd(run_path):
                try:
                    # compile the test
                    rr = su.bash.run(f"javac -cp {classpath} {test_path}")
                    #
                    rr = su.bash.run(
                        f"java -cp .:{Tool.rt_jar}:{classpath} -ea org.junit.runner.JUnitCore {test_method.cname}",
                        timeout=10,
                    )

                    if ex_log.exists():
                        called_methods = su.io.load(ex_log, su.io.Fmt.txt)
                        cover_stmt = extract_covered_throw_from_stack_trace(
                            called_methods
                        )
                        if (
                            stack_trace[-1][0]["method"].split("#")[0] == cover_stmt[0]
                            and stack_trace[-1][0]["method"].split("#")[1]
                            == cover_stmt[1]
                            and stack_trace[-1][1] + stack_trace[-1][0]["startLine"]
                            == int(cover_stmt[2])
                        ):
                            coverage = 1
                        if coverage == 0:
                            print("Generated etest does not cover the target statement")
                            print(cover_stmt)
                except:
                    breakpoint()
                    logger.warning("Failed to run the the test!")
        except:
            exc_tr = traceback.format_exc()
            logger.warning(f"Failed to compile the test! {exc_tr}")
        return coverage

    # helper functions ====
    def compare_two_models_results(self, model_a: str, model_b: str):
        model_a_results_dir = (
            Macros.exp_dir / model_a / "lora-codellama-7b" / "test-results"
        )
        model_b_results_dir = (
            Macros.exp_dir / model_b / "lora-codellama-7b" / "test-results"
        )

        model_a_results = load_dataset(model_a_results_dir, clz=LLMResults)
        model_b_results = load_dataset(model_b_results_dir, clz=LLMResults)
        for a_res, b_res in zip(model_a_results, model_b_results):
            if a_res.metrics["coverage-max"] > 0 and b_res.metrics["coverage-max"] < 1:
                print(a_res.id)

    def temp_eval(self, selected_ids: List):

        llm_preds: List[LLMResults] = self._load_preds()

        runtime_metrics = []
        for pred in llm_preds:
            if pred.id in selected_ids:
                runtime_metrics.append(pred.metrics)
        metrics_summary = summarize_metrics(aggregate_metrics(runtime_metrics))
        print(metrics_summary)


def extract_covered_throw_from_stack_trace(raw_stack_traces: str):
    """
    Extract the last called method and the line number.
    """
    raw_stack_traces = raw_stack_traces.split("##")
    raw_stack_traces = list(filter(lambda x: x != "", raw_stack_traces))

    stack_trace_frame = raw_stack_traces[0]
    class_name = stack_trace_frame.split("#")[0]
    method_name = stack_trace_frame.split("#")[1]
    line_number = stack_trace_frame.split("#")[2]

    return class_name, method_name, line_number


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(LLMCodeEvaluator, as_positional=False)
