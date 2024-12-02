import seutil as su
from tqdm import tqdm
from typing import *
import random
from thefuzz import fuzz

random.seed(42)
from jsonargparse import CLI
from pathlib import Path

from etestgen.macros import Macros
from etestgen.data.utils import load_dataset
from etestgen.collector.MetricsCollector import remove_duplicate_stack_trace
from etestgen.llm.prompts import (
    add_called_comment_to_method,
    format_stack_trace_prompt,
)
from etestgen.data.data import DataNE2E, parse_data_cls
from etestgen.llm.prompts import (
    add_called_comment_to_method,
)


class DataProcessor:
    def __init__(self, config_file: Path):
        self.config = su.io.load(config_file)
        self.system_message = self.config["system_message"]
        self.data_dir = Macros.work_dir / "setup" / self.config["setup"]
        self.data_type = self.config["data_type"]
        self.setup = self.config["setup"]

    def process_test_data(self):
        """
        Process model input and output that suit for CodeLLaMA.
        """

        test_dataset_list = []
        test_dataset = self.load_test_data()
        with tqdm(
            total=len(test_dataset),
            desc="Processing real test data for CodeLLaMA",
        ) as pbar:
            for dt in test_dataset:
                data_list = self.format_data_for_llama(self.data_type)(dt)
                data_id = dt.id
                test_dataset_list.extend(
                    [
                        {"id": data_id, "instruction": input, "output": "```java\nNone\n```\n</s>"}
                        for input, output in data_list
                    ]
                )
                pbar.update(1)

        su.io.dump(
            Macros.work_dir / self.config["test_data"] / f"test-{self.setup}.jsonl",
            test_dataset_list,
        )

    def load_test_data(self):
        data_dir = Macros.work_dir / self.config["test_data"]
        with tqdm("Loading test data") as pbar:
            test_dataset = load_dataset(
                data_dir,
                clz=parse_data_cls(self.config["data_type"]),
                pbar=pbar,
            )
        return test_dataset

    def format_data_for_llama(self, which: str):
        if which == "MUT2E":
            return self.format_Llama_input_from_mut2e
        elif which == "NE2E":
            return self.format_Llama_input_from_ne2e
        elif which == "NE2E-ALL":
            return self.format_Llama_input_from_ne2e_all
        elif which == "CONDITIONNESTACK2E":
            return self.format_Llama_input_from_conditionnestack2e
        elif which == "CONDITIONNESTACK2E-ALL":
            return self.format_Llama_input_from_conditionnestack2e_all
        else:
            raise ValueError(f"Unknown data type: {which}")

    def format_Llama_input_from_mut2e(self, dt: Any) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for CodeLLaMA.
        Outputting prompt include MUT
        """

        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split(".")[-1]
        exception_type = dt.etype.split(".")[-1]
        etest_context = get_test_context(dt)
        mut_code = dt.mut
        dt.e_stack_trace = dt.e_stack_trace[0]
        mut_code = add_called_comment_to_method(dt.mut, line_num=dt.e_stack_trace[0][1])
        # etest_class = dt.test_e_key.split("#")[0].split("/")[-1]
        instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
        input = (
            f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
        )
        output = f"""```java\n{dt.test_e}\n```\n</s>"""

        return [(input, output)]

    def format_Llama_input_from_ne2e(self, dt: Any) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for ne2e data for CodeLLaMA.
        Outputting prompt include MUT & related NEBT (if exist)
        """

        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split(".")[-1]
        exception_type = dt.etype.split(".")[-1]
        mut_code = dt.mut
        mut_code = add_called_comment_to_method(dt.mut, line_num=dt.e_stack_trace[0][1])
        etest_context = get_test_context(dt)
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        paired_data = []
        if not dt.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            output = f"""```java\n{dt.test_e}\n```\n</s>"""
            paired_data.append((input, output))
        else:
            ne_test = random.choice(dt.test_ne)
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_ne2e_all(self, dt: Any) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for ne2e-all data for CodeLLaMA. Try at most 5 netest for generation.
        """

        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = dt.etype.split(".")[-1]
        mut_code = dt.mut
        mut_code = add_called_comment_to_method(dt.mut, line_num=dt.e_stack_trace[0][1])
        etest_context = get_test_context(dt)
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        paired_data = []
        if not dt.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            output = f"""```java\n{dt.test_e}\n```\n</s>"""
            paired_data.append((input, output))
        else:
            sample_netests = random.sample(list(dt.test_ne), min(5, len(dt.test_ne)))
            for ne_test in sample_netests:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_conditionnestack2e_all(
        self, dt: Any
    ) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for conditionne2e with all possible netest data for CodeLLaMA.\
        Outputting prompt include MUT & related NEBT (if exist) & condition (if exist)
        """

        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = dt.etype.split(".")[-1]
        mut_code = dt.mut
        dt.e_stack_trace = dt.e_stack_trace[0]
        mut_code = add_called_comment_to_method(dt.mut, dt.e_stack_trace[0][1])
        etest_context = get_test_context(dt)
        condition = dt.condition
        if len(condition) > 0 and condition != "":
            condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition[0]}\n```\n"
        else:
            condition_prompt = ""
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        paired_data = []
        if not dt.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            output = f"""```java\n{dt.test_e}\n```\n</s>"""
            paired_data.append((input, output))
        else:
            for ne_test in dt.test_ne:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_conditionnestack2e(self, dt: Any) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for conditionne2e data for CodeLLaMA.
        """

        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = dt.etype.split(".")[-1]
        mut_code = dt.mut
        dt.e_stack_trace = dt.e_stack_trace[0]
        mut_code = add_called_comment_to_method(dt.mut, dt.e_stack_trace[0][1])
        etest_context = get_test_context(dt)
        condition = dt.condition
        if len(condition) > 0 and condition != "":
            condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition[0]}\n```\n"
        else:
            condition_prompt = ""
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        paired_data = []
        if not dt.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            output = f"""```java\n{dt.test_e}\n```\n</s>"""
            paired_data.append((input, output))
        else:
            ne_test = random.choice(dt.test_ne)
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            paired_data.append((input, output))
        return paired_data

    def select_shortest_stack_trace(self, dt: Any):
        """Return the shortest stack trace"""
        stack_traces = dt.stack_traces
        shortest_stack_trace = stack_traces[0]
        for stack_trace in stack_traces:
            if len(stack_trace) < len(shortest_stack_trace):
                shortest_stack_trace = stack_trace
        dt.call_stacks = shortest_stack_trace
        return dt


def find_the_most_revelant_netest(dt: Any) -> Tuple[str]:
    """
    Given MUT, find the most relevant netest.
    1. search by pattern matching
    2. fuzz matching
    Return netest, and context
    """
    mut_class_name = dt.mut_key.split("#")[0].split("/")[-1]
    mut_method_name = dt.mut_key.split("#")[1]
    best_netest_method, best_netest, test_context, best_score = None, None, None, -1
    for ne_test, ne_test_key in zip(dt.test_ne, dt.test_ne_key):
        ne_test_class = ne_test_key.split("#")[0].split("/")[-1]
        ne_test_method = ne_test_key.split("#")[1]
        if (
            f"Test{mut_class_name}" == ne_test_class
            or f"{mut_class_name}Test" == ne_test_class
        ):
            best_netest = ne_test
            mname_fuzz_ratio = fuzz.ratio(mut_method_name, ne_test_method)
            fuzz_ratio = mname_fuzz_ratio + 100
        else:
            fuzz_ratio = fuzz.ratio(mut_class_name, ne_test_class)
        if fuzz_ratio > best_score:
            best_netest = ne_test
            best_score = fuzz_ratio
    return best_netest


def get_test_context(dt: DataNE2E) -> str:
    if type(dt.test_context) == list:
        etest_context = (
            dt.test_context[0].replace("adhoc_", "").replace("/*TEST PLACEHOLDER*/", "")
        )
    elif dt.test_context is not None:
        etest_context = dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
    else:
        etest_context = "import static org.junit.Assert.*;\nimport org.junit.Test;\n\npublic class Test {\n\n}\n"
    return etest_context


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(DataProcessor, as_positional=False)
