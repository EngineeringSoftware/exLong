from typing import Any, List, Union
import seutil as su
from jsonargparse import CLI
from tqdm import tqdm
import random

from etestgen.data.utils import load_dataset
from etestgen.llm.utils import LLMResults
from etestgen.llm.prompts import add_called_comment_to_method
from etestgen.macros import Macros

random.seed(42)


class PredictionInspector:
    def __init__(self, config_file: Union[str, su.arg.RPath], eval_set: str = "rq2"):
        self.config = su.io.load(config_file)
        self.eval_set = eval_set
        self.setup = self.config["setup"]
        self.exp = self.config["model_name"]
        self.prediction_dir = (
            Macros.exp_dir / self.setup / self.exp / f"{eval_set}-results"
        )

    def find_predictions_for_potential_pull_request(self):
        """
        Find the predictions by LLM that we could submit pull request.
        """

        already_check_projects = [
            "arquillian_arquillian-core",
            "opencb_java-common-libs",
            "OpenNMS_newts",
            "greenmail-mail-test_greenmail",
            "OpenHFT_Chronicle-Map",
            "analogweb_core",
            "JodaOrg_joda-beans",
            "Coreoz_Wisp",
        ]
        to_check = [
            "javadev_moneytostr-russian",
            "microfocus-idol_java-content-parameter-api",
            "spotify_async-google-pubsub-client",
        ]
        llm_preds = self.load_predictions()
        project_to_stars = self.project_to_stars()

        reorder_examples = []
        for project_name in project_to_stars:
            if project_name not in to_check:
                continue
            for pred in llm_preds:
                if pred.project == project_name and pred.metrics["coverage-max"] > 0:
                    reorder_examples.append(pred)
        #
        examples_to_write = []
        for llm_res in reorder_examples:
            example_markdown = self.rq2_format(llm_res)
            examples_to_write.append(example_markdown)
        su.io.dump(
            Macros.doc_dir / f"LLM-predictions-on-pull-request.md",
            "\n\n".join(examples_to_write),
            su.io.Fmt.txt,
        )

    def find_model_predicted_etest_for_project(self, target_projects: List[str]):
        llm_preds = self.load_predictions()
        # reorder the predictions by projects
        reorder_examples = []
        for project_name in target_projects:
            for pred in llm_preds:
                if pred.project == project_name and pred.metrics["coverage-max"] > 0:
                    reorder_examples.append(pred)
        #
        examples_to_write = []
        for llm_res in reorder_examples:
            example_markdown = self.rq2_format(llm_res)
            examples_to_write.append(example_markdown)
        su.io.dump(
            Macros.doc_dir / f"LLM-prediction-on-potential-pull-request.md",
            "\n\n".join(examples_to_write),
            su.io.Fmt.txt,
        )

    def write_model_unique_predictions_to_file(self):
        """
        Find the covered stmts that are unique to the model's prediction compared to EvoSuite and Randoop.
        """
        model_covered_throw_stmts = su.io.load(
            Macros.results_dir
            / "tool-results"
            / f"{self.setup}-{self.eval_set}-covered-stmts.json"
        )
        evosuite_covered_throw_stmts_2_tests = su.io.load(
            Macros.results_dir
            / "tool-results"
            / f"evosuite-{self.eval_set}-all-covered-stmts-2-tests.json"
        )
        randoop_covered_throw_stmts_2_tests = su.io.load(
            Macros.results_dir
            / "tool-results"
            / f"randoop-{self.eval_set}-all-covered-stmts-2-tests.json"
        )
        # first find only model is able to generate
        evosute_covered_stmts = set(evosuite_covered_throw_stmts_2_tests.keys())
        randoop_covered_stmts = set(randoop_covered_throw_stmts_2_tests.keys())
        model_unique_covered_throw_stmts = (
            set(model_covered_throw_stmts)
            - evosute_covered_stmts
            - randoop_covered_stmts
        )
        self.write_model_preds_to_file(list(model_unique_covered_throw_stmts))
        # find only evosuite is able to generate
        evosuite_unique_covered_throw_stmts = (
            evosute_covered_stmts
            - set(model_covered_throw_stmts)
            - randoop_covered_stmts
        )
        evo_test_to_check = [
            {s: evosuite_covered_throw_stmts_2_tests[s]}
            for s in evosuite_unique_covered_throw_stmts
        ]
        su.io.dump(
            Macros.results_dir / "tool-results" / "evosuite-unique-covered-tests.json",
            evo_test_to_check,
            su.io.Fmt.json,
        )
        # find only randoop is able to generate
        randoop_unique_covered_throw_stmts = (
            randoop_covered_stmts
            - set(model_covered_throw_stmts)
            - evosute_covered_stmts
        )
        randoop_test_to_check = [
            {s: randoop_covered_throw_stmts_2_tests[s]}
            for s in randoop_unique_covered_throw_stmts
        ]
        su.io.dump(
            Macros.results_dir / "tool-results" / "randoop-unique-covered-tests.json",
            randoop_test_to_check,
            su.io.Fmt.json,
        )

    def write_model_predictions_with_netest(self, sample_size: int = 10):
        # load LLMresults
        llm_preds = self.load_predictions()

        examples_to_write = []
        random.shuffle(llm_preds)
        for llm_res in llm_preds:
            llm_metrics = llm_res.metrics
            if "coverage-max" not in llm_metrics:
                continue
            if llm_metrics["coverage-max"] > llm_metrics["coverage-min"]:
                # write to markdown file to
                example_markdown = self.rq2_format(llm_res)
                examples_to_write.append(example_markdown)
            if len(examples_to_write) >= sample_size:
                break
        su.io.dump(
            Macros.doc_dir / f"LLM-prediction-on-different-netests.md",
            "\n\n".join(examples_to_write),
            su.io.Fmt.txt,
        )

    def write_model_preds_to_file(self, data_to_check: List[str]):
        """
        Sample examples from model's prediction for inspection.
        data_to_check: list of covered stmts to check.
        """
        llm_preds = self.load_predictions()

        pred_to_check = []
        for pred in llm_preds:
            stack_trace = pred.input.e_stack_trace[0]
            line_number = stack_trace[-1][1] + stack_trace[-1][0]["startLine"]
            project_name = pred.project
            mut_key = "#".join(pred.input.mut_key.split("#")[:2])
            key = f"{project_name}#{mut_key}#{line_number}"
            if key in data_to_check:
                pred_to_check.append(pred)

        examples_to_write = []
        for data in pred_to_check:
            example_markdown = self.rq2_format(data)
            examples_to_write.append(example_markdown)

        su.io.dump(
            Macros.doc_dir
            / f"model-predictions-{self.setup}-{self.exp}-{self.eval_set}-examples.md",
            "\n\n".join(examples_to_write),
            su.io.Fmt.txt,
        )

    def val_data_inspect_format(self, data: LLMResults) -> str:
        # prepare for content
        src_input = data.input
        mut_code = add_called_comment_to_method(
            src_input.mut, src_input.e_stack_trace[0][0][1]
        )
        etest_context = get_test_context(src_input)
        if src_input.condition:
            condition = src_input.condition
        else:
            # condition_pt = (
            #     "The following condition should be satisfied to trigger the exception:"
            # )
            condition = [""]
        s = f"# data #{data.id}\n"
        s += f"## proj_name: {data.project}\n"
        s += f"- **mut**: {src_input.mut_key}\n"
        mut_class_name = src_input.mut_key.split("#")[0].split("/")[-1]
        s += "```java\n"
        s += f"public class {mut_class_name}"
        s += "{\n"
        s += mut_code
        s += "\n}\n"
        s += "\n```\n"
        s += f"- **exception type**: {src_input.etype}\n"
        s += f"- **condition**: {condition}\n"
        if data.gold and data.gold != "":
            s += f"- **gold**:\n```java\n{data.gold}\n```\n"
        assert len(data.topk) == len(data.prompt)
        for i, pred in enumerate(data.topk):
            s += f"## prompt {i}\n"
            s += data.prompt[i] + "\n"
            test_file_content = pred

            s += f"## prediction {i}\n"
            s += "```java\n"
            s += test_file_content
            s += "\n```\n"
            s += f"## Metrics:\n"
            s += f"compilable: "
            s += str(data.metrics["compilable-topk"][i]) + "\n"
            s += f"runnable: "
            runnable_rate = data.metrics["runnable-overall-topk"][i]
            s += str(runnable_rate) + "\n"
            s += f"coverage: "
            s += str(data.metrics["coverage-topk"][i]) + "\n"

            s += "\n"
        return s

    def rq2_format(self, data: LLMResults) -> str:
        # prepare for content
        src_input = data.input
        mut_code = add_called_comment_to_method(
            src_input.mut, src_input.e_stack_trace[0][0][1]
        )
        if src_input.condition:
            condition = src_input.condition
        else:
            # condition_pt = (
            #     "The following condition should be satisfied to trigger the exception:"
            # )
            condition = [""]
        etest_context = get_test_context(src_input)
        s = f"# data #{data.id}\n"
        s += f"## proj_name: {data.project}\n"
        s += f"- **mut**: {src_input.mut_key}\n"
        mut_class_name = src_input.mut_key.split("#")[0].split("/")[-1]
        s += "```java\n"
        s += f"public class {mut_class_name}"
        s += "{\n"
        s += mut_code
        s += "\n}\n"
        s += "\n```\n"
        s += f"- **exception type**: {src_input.etype}\n"
        s += f"- **condition**: {condition}\n"
        s += f"- **etest context**:\n"
        s += "```java\n"
        s += etest_context
        s += "\n```\n"
        for i, pred in enumerate(data.topk):
            test_file_content = pred
            s += f"## prediction {i}\n"
            s += "```java\n"
            s += test_file_content
            s += "\n```\n"
            s += "\n"
        return s

    def load_predictions(self):
        llm_outputs = load_dataset(save_dir=self.prediction_dir, clz=LLMResults)
        return llm_outputs

    def project_to_stars(self):
        project_list = su.io.load(Macros.work_dir / "repos" / "filtered" / "repos.json")
        project_to_stars = {}
        for p in project_list:
            project_to_stars[p["full_name"]] = p["stars"]
        # sort project based on stars
        project_to_stars = dict(
            sorted(project_to_stars.items(), key=lambda item: item[1], reverse=True)
        )
        return project_to_stars


def get_test_context(dt) -> str:
    if type(dt.test_context) == list:
        etest_context = dt.test_context[0].replace("adhoc_", "")
    elif dt.test_context is not None:
        etest_context = dt.test_context.replace("adhoc_", "")
    else:
        etest_context = "import static org.junit.Assert.*;\nimport org.junit.Test;\n\npublic class Test {\n\n}\n"
    return etest_context


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(PredictionInspector, as_positional=False)
