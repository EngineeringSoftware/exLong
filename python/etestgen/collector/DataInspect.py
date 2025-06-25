from typing import Any, List
import seutil as su
from jsonargparse import CLI
import random

from etestgen.data.utils import load_dataset
from etestgen.llm.utils import LLMResults
from etestgen.llm.prompts import add_called_line_to_method, add_called_comment_to_method
from etestgen.macros import Macros
from etestgen.data.utils import save_dataset
from etestgen.data.data import DataNE2E

random.seed(42)


class DataInspector:
    def __init__(self):
        self.output_dir = Macros.doc_dir

    def reorganize_rq2_public(self):
        rq2_public = self.open_dataset(dataset_name="rq2-public")
        new_dataset = []
        for rq2_data in rq2_public:
            rq2_data.e_stack_trace = [rq2_data.e_stack_trace]
            new_dataset.append(rq2_data)

        save_dataset(Macros.data_dir / "rq2-public-new", new_dataset, clz=DataNE2E)

    def write_dataset_to_markdown(self, dataset_name: str):

        # load rq2 public methods
        dataset = self.open_dataset(dataset_name=dataset_name)

        examples_to_write = []
        for data in dataset:
            data_example = self.datane2e_pformat(data)
            examples_to_write.append(data_example)
            # if len(examples_to_write) > sample_size:
            #     break

        su.io.dump(
            self.output_dir / f"dataset-{dataset_name}-examples.md",
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
            Macros.doc_dir / f"{self.setup}-{self.exp}-{self.eval_set}-examples.md",
            "\n\n".join(examples_to_write),
            su.io.Fmt.txt,
        )

    def rq2_format(self, data: LLMResults) -> str:
        # prepare for content
        src_input = data.input
        mut_code = add_called_comment_to_method(
            src_input.mut, src_input.e_stack_trace[0][0][1]
        )
        etest_context = get_test_context(src_input)
        if src_input.condition:
            condition = src_input.condition
        else:
            condition_pt = (
                "The following condition should be satisfied to trigger the exception:"
            )
            condition = data.prompt[0].split(condition_pt)[1].split("```")[0]
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
        s += f"- **condition**: {condition[0]}\n"
        if data.gold and data.gold != "":
            s += f"- **gold**:\n {data.gold}\n"
        for i, pred in enumerate(data.topk):
            if src_input.test_ne:
                s += f"- **ne-test**: \n```java\n{src_input.test_ne[i]}\n```\n"
            if i == 0:
                test_file_content = etest_context.replace(
                    "/*TEST PLACEHOLDER*/", "\n" + pred
                )
            else:
                test_file_content = pred

            s += f"## prediction {i}\n"
            s += "```java\n"
            s += test_file_content
            s += "\n```\n"
            s += "\n"
        return s

    def datane2e_pformat(self, data: DataNE2E) -> str:
        mut_code = add_called_comment_to_method(data.mut, data.e_stack_trace[0][0][1])

        s = f"# data #{data.id}\n"
        s += f"## proj_name: {data.project}\n"
        s += f"- **mut**: {data.mut_key}\n"
        mut_class_name = data.mut_key.split("#")[0].split("/")[-1]
        if data.condition:
            s += f"- **condition**: {data.condition}\n"
        s += "```java\n"
        s += f"public class {mut_class_name}"
        s += "{\n"
        s += mut_code
        s += "\n}\n"
        s += "\n```\n"
        s += f"- **exception type**: {data.etype}\n"
        s += "\n"
        return s

    def inspect_stackne2e(self, id: int, with_source=False) -> str:
        """
        Format the model prediction for easy inspection.
        """
        data = self.dataset[id]
        # assert data.id == pred.data_id

        s = f"## data #{data.id}\n"
        s += f"### proj_name: {data.project}\n"

        s = f"# data #{id}\n"
        s += f"# proj_name: {data.project}\n"
        s += f"# exception test: {data.test_e_key}\n"
        s += f"# **mut**: {data.mut_key}\n"
        s += "```java\n"
        s += data.mut
        s += "\n```\n"
        s += f"- **exception type**: {data.etype}\n"
        s += f"# stacktrace of thrown exception method\n"
        for i, called_method in enumerate(data.call_stacks):
            s += f"```java\n"
            # s += f"// call stack no. {i}\n"
            s += f"// {called_method.namedesc}\n"
            s += f"\n {called_method.code} \n"
            s += "// $$$$$$$$$$$$$$$$$$$$$\n\n"
            # for m in c_stacks:
            #     s += f"{m}\n\n"
            s += f"```\n"
        s += "\n"
        return s

    def load_predictions(self):
        llm_outputs = load_dataset(save_dir=self.prediction_dir, clz=LLMResults)
        return llm_outputs

    def open_dataset(self, dataset_name: str):
        llm_outputs = load_dataset(
            save_dir=Macros.data_dir / dataset_name, clz=DataNE2E
        )
        return llm_outputs


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
    CLI(DataInspector, as_positional=False)
