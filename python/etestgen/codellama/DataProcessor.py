import seutil as su
from tqdm import tqdm
from typing import *
import random
from thefuzz import fuzz

random.seed(42)
from jsonargparse import CLI
from pathlib import Path
from transformers import (
    AutoTokenizer,
)

from etestgen.macros import Macros
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.collector.MetricsCollector import remove_duplicate_stack_trace
from etestgen.llm.prompts import (
    add_called_comment_to_method,
    format_stack_trace_prompt,
)
from etestgen.data.data import parse_data_cls
from etestgen.llm.prompts import (
    LLM_stack_trace_prompt,
    add_called_comment_to_method,
    format_stack_trace_prompt,
)


class DataProcessor:
    def __init__(self, config_file: Path):
        self.config = su.io.load(config_file)
        self.system_message = self.config["system_message"]
        self.data_dir = Macros.work_dir / "setup" / self.config["setup"]
        self.data_type = self.config["data_type"]
        self.setup = self.config["setup"]

    def process_train_data(self):
        """
        Process model input and output that suit for CodeLLaMA.
        """

        train_dataset_list, val_dataset_list = [], []
        train_dataset, val_dataset = self.load_train_data()
        with tqdm(
            total=len(train_dataset) + len(val_dataset),
            desc="Processing training data for CodeLLaMA",
        ) as pbar:
            for mut_dt in train_dataset:
                data_list = self.format_data_for_llama(self.data_type)(mut_dt)
                train_dataset_list.extend(
                    [
                        {"instruction": input, "output": output}
                        for input, output in data_list
                    ]
                )
                pbar.update(1)
            for mut_dt in val_dataset:
                data_list = self.format_data_for_llama(self.data_type)(mut_dt)
                val_dataset_list.extend(
                    [
                        {"instruction": input, "output": output}
                        for input, output in data_list
                    ]
                )
                train_dataset_list.extend(
                    [
                        {"instruction": input, "output": output}
                        for input, output in data_list
                    ]
                )
                pbar.update(1)
        su.io.mkdir(self.data_dir / "train" / "train")
        su.io.dump(
            self.data_dir / "train" / "train" / f"train-{self.setup}.jsonl",
            train_dataset_list,
        )

        su.io.dump(
            self.data_dir / "train" / "val" / f"val-{self.setup}.jsonl",
            val_dataset_list,
        )

    def process_test_data(self):
        """
        Process model input and output that suit for CodeLLaMA.
        """

        test_dataset_list = []
        test_dataset = self.load_test_data()
        with tqdm(
            total=len(test_dataset),
            desc="Processing test data for CodeLLaMA",
        ) as pbar:
            for mut_dt in test_dataset:
                data_list = self.format_data_for_llama(self.data_type)(mut_dt)
                data_id = mut_dt.id
                test_dataset_list.extend(
                    [
                        {"id": data_id, "instruction": input, "output": output, "project_name": mut_dt.project}
                        for input, output in data_list
                    ]
                )
                pbar.update(1)

        su.io.dump(
            self.data_dir / "eval" / "test" / f"test-{self.setup}.jsonl",
            test_dataset_list,
        )

    def process_val_data(self):
        """
        Process model input and output for validation set (this may be applied to prediction inspection.)
        """

        test_dataset_list = []
        val_dataset = self.load_val_data()
        with tqdm(
            total=len(val_dataset),
            desc="Processing test data for CodeLLaMA",
        ) as pbar:
            # special filter
            for mut_dt in val_dataset:
                if not (mut_dt.test_ne and len(mut_dt.test_ne) > 1):
                    continue
                data_list = self.format_data_for_llama(self.data_type)(mut_dt)
                data_id = mut_dt.id
                test_dataset_list.extend(
                    [
                        {"id": data_id, "instruction": input, "output": output}
                        for input, output in data_list
                    ]
                )
                pbar.update(1)

        su.io.dump(
            self.data_dir / "eval" / "val" / f"val-{self.setup}.jsonl",
            test_dataset_list,
        )

    def process_real_test_data(self):
        """
        Process model input and output that suit for CodeLLaMA for real test set.
        """

        test_dataset_list = []
        test_dataset = self.load_real_test_data()
        with tqdm(
            total=len(test_dataset),
            desc="Processing test data for CodeLLaMA",
        ) as pbar:
            for mut_dt in test_dataset:
                data_id = mut_dt.id
                data_list = self.format_data_for_llama(f"{self.data_type}")(mut_dt)
                test_dataset_list.extend(
                    [
                        {"id": data_id, "instruction": input, "output": output, "project_name": mut_dt.project}
                        for input, output in data_list
                    ]
                )
                pbar.update(1)

        su.io.dump(
            self.data_dir / "real-eval" / "test" / f"test-{self.setup}.jsonl",
            test_dataset_list,
        )

    ## Helper functions
    ###
    def compute_generation_tokens(self):
        train_dataset, val_dataset = self.load_train_data()
        test_dataset = self.load_test_data()
        inspect_dataset = [train_dataset, test_dataset]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model_name"], trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        for dtset in inspect_dataset:
            train_dataset_list = []
            train_data_length = []
            input_data, output_data = [], []
            input_length, output_length = [], []
            with tqdm(
                total=len(dtset),
                desc="Calculating sequence length of the dataset for CodeLLaMA",
            ) as pbar:
                for mut_dt in dtset:
                    data_list = self.format_data_for_llama(self.data_type)(mut_dt)
                    train_dataset_list.extend(
                        [f"{input} {output}" for input, output in data_list]
                    )
                    input_data.extend([input for input, output in data_list])
                    output_data.extend([output for input, output in data_list])
                    pbar.update(1)

            for data_seq, input_seq, output_seq in zip(
                train_dataset_list, input_data, output_data
            ):
                train_data_length.append(
                    len(
                        self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(data_seq)
                        )
                    )
                )
                input_length.append(
                    len(
                        self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(input_seq)
                        )
                    )
                )
                output_length.append(
                    len(
                        self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(output_seq)
                        )
                    )
                )

            for data_length in [train_data_length, input_length, output_length]:
                data_length.sort()
                total_count = len(data_length)
                top_10_index = int(0.1 * total_count)
                seq_top_10_percent_value = data_length[-top_10_index]
                # Find the values at the calculated indices
                print(seq_top_10_percent_value)

    def load_train_data(self):
        data_dir = self.data_dir / "train"
        with tqdm("Loading training data") as pbar:
            train_dataset = load_dataset(
                data_dir / "train",
                clz=parse_data_cls(self.config["data_type"]),
                pbar=pbar,
            )
        with tqdm("Loading validation data") as pbar:
            val_dataset = load_dataset(
                data_dir / "val",
                clz=parse_data_cls(self.config["data_type"]),
                pbar=pbar,
            )
        return train_dataset, val_dataset

    def load_val_data(self):
        data_dir = self.data_dir / "eval"
        with tqdm("Loading validation data") as pbar:
            val_dataset = load_dataset(
                data_dir / "val",
                clz=parse_data_cls(self.config["data_type"]),
                pbar=pbar,
            )
        return val_dataset

    def load_test_data(self):
        data_dir = Macros.work_dir / self.config["test_data"]
        with tqdm("Loading test data") as pbar:
            test_dataset = load_dataset(
                data_dir,
                clz=parse_data_cls(self.config["data_type"]),
                pbar=pbar,
            )
        return test_dataset

    def load_real_test_data(self):
        data_dir = self.data_dir / "real-eval"
        with tqdm("Loading real test data") as pbar:
            test_dataset = load_dataset(
                data_dir / "test",
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
        elif which == "NESTACK2E":
            return self.format_Llama_input_from_true_nestack2e
        elif which == "NESTACK2E-ALL":
            return self.format_Llama_input_from_true_nestack2e_all
        elif which == "CONDITIONNE2E":
            return self.format_Llama_input_from_conditionne2e
        elif which == "CONDITIONNESTACK2E-SAMPLE":
            return self.format_Llama_input_from_conditionnestack2e_sample
        elif which == "CONDITIONNESTACK2E":
            return self.format_Llama_input_from_conditionnestack2e
        elif which == "CONDITIONNESTACK2E-ALL":
            return self.format_Llama_input_from_conditionnestack2e_all
        elif which == "CONDITIONNESTACK2E-REAL":
            return self.format_Llama_input_from_conditionnestack2e_real
        else:
            raise ValueError(f"Unknown data type: {which}")

    def format_Llama_input_from_mut2e(self, mut_dt: Any) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for CodeLLaMA.
        """

        mut_name = mut_dt.mut_key.split("#")[1]
        mut_class = mut_dt.mut_key.split("#")[0].split(".")[-1]
        exception_type = mut_dt.etype.split(".")[-1]
        etest_context = mut_dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        mut_code = mut_dt.mut
        etest_class = mut_dt.test_e_key.split("#")[0].split("/")[-1]
        etest_name = mut_dt.test_e_key.split("#")[1]
        if self.config["with_name"]:
            instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
        else:
            instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
        input = (
            f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
        )
        output = f"""```java\n{mut_dt.test_e}\n```\n</s>"""

        return [(input, output)]

    def format_Llama_input_from_ne2e(self, ne2e_dt: Any) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for ne2e data for CodeLLaMA.
        """

        mut_name = ne2e_dt.mut_key.split("#")[1]
        mut_class = ne2e_dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = ne2e_dt.etype.split(".")[-1]
        mut_code = ne2e_dt.mut
        etest_name = ne2e_dt.test_e_key.split("#")[1]
        etest_context = ne2e_dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        output = f"""```java\n{ne2e_dt.test_e}\n```\n</s>"""
        paired_data = []
        if not ne2e_dt.test_ne:
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{ne2e_dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{ne2e_dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
        else:
            ne_test = random.choice(ne2e_dt.test_ne)
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((input, output))
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_ne2e_all(self, ne2e_dt: Any) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for ne2e-all data for CodeLLaMA. Try at most 5 netest for generation.
        """

        mut_name = ne2e_dt.mut_key.split("#")[1]
        mut_class = ne2e_dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = ne2e_dt.etype.split(".")[-1]
        mut_code = ne2e_dt.mut
        etest_name = ne2e_dt.test_e_key.split("#")[1]
        etest_context = ne2e_dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        output = f"""```java\n{ne2e_dt.test_e}\n```\n</s>"""
        paired_data = []
        if not ne2e_dt.test_ne:
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{ne2e_dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{ne2e_dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
        else:
            sample_netests = random.sample(
                list(ne2e_dt.test_ne), min(5, len(ne2e_dt.test_ne))
            )
            for ne_test in sample_netests:
                if self.config["with_name"]:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                    paired_data.append((input, output))
                else:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                    paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_conditionnestack2e_real(self, dt: Any):
        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = dt.etype.split(".")[-1]
        mut_code = dt.mut
        etest_name = dt.test_e_key.split("#")[1]
        etest_class = dt.test_e_key.split("#")[0].split("/")[-1]
        etest_context = dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        dt.condition = [dt.condition]
        dt.e_stack_trace = [dt.e_stack_trace[0]]
        assert len(dt.e_stack_trace) == len(
            dt.condition
        ), "stack trace and condition should have the same length"
        condition_prompts = []
        stack_trace_prompts = []

        for stack_trace, condition in zip(dt.e_stack_trace, [dt.condition]):
            # process stack trace:
            if len(stack_trace) > 6:
                stack_trace = remove_duplicate_stack_trace(stack_trace)
            stack_trace = [stack_trace]
            mut_code = add_called_comment_to_method(dt.mut, line_num=stack_trace[0][1])
            stack_trace_prompt = LLM_stack_trace_prompt(
                dt, stack_trace, ignore_num=1000
            )
            stack_trace_prompts.append(stack_trace_prompt)
            if condition != "":
                condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition}\n```\n"
            else:
                condition_prompt = ""
            condition_prompts.append(condition_prompt)
        assert len(stack_trace_prompts) == len(
            condition_prompts
        ), "stack trace and condition should have the same length"
        sample_condition_prompts = condition_prompts
        sample_stack_trace_prompts = stack_trace_prompts
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        if len(sample_condition_prompts) == 0:
            assert len(sample_stack_trace_prompts) == 0
            sample_condition_prompts = [""]
            sample_stack_trace_prompts = [""]
        paired_data = []
        for condition_prompt, stack_trace_prompt in zip(
            sample_condition_prompts, sample_stack_trace_prompts
        ):
            if not dt.test_ne:
                if self.config["with_name"]:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class {etest_class}. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                    output = f"""```java\n{dt.test_e}\n```\n</s>"""
                    paired_data.append((input, output))
                else:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Please only give the new exceptional-behavior test method to complete the following test class {etest_class}. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                    output = f"""```java\n{dt.test_e}\n```\n</s>"""
                    paired_data.append((input, output))
            else:
                ne_test = random.choice(dt.test_ne)
                if self.config["with_name"]:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class  {etest_class}. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                else:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class  {etest_class}. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_conditionne2e(self, dt: Any) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for conditionne2e data for CodeLLaMA.
        """

        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = dt.etype.split(".")[-1]
        mut_code = dt.mut
        mut_line = dt.e_stack_trace[0][1]
        mut_code = add_called_comment_to_method(mut_code, mut_line)
        etest_name = dt.test_e_key.split("#")[1]
        etest_context = dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        condition = dt.condition
        if condition and condition != "":
            condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition}\n```\n"
        else:
            condition_prompt = ""
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        paired_data = []
        if not dt.test_ne:
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Please only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
        else:
            ne_test = random.choice(dt.test_ne)
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
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
        if dt.e_stack_trace:
            mut_code = add_called_comment_to_method(dt.mut, dt.e_stack_trace[0][1])
        etest_name = dt.test_e_key.split("#")[1]
        etest_context = dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        stack_trace_prompt = LLM_stack_trace_prompt(dt, dt.e_stack_trace)
        condition = dt.condition
        if condition and condition != "":
            condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition}\n```\n"
        else:
            condition_prompt = ""
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        paired_data = []
        if not dt.test_ne:
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Please only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
        else:
            ne_test = random.choice(dt.test_ne)
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_conditionnestack2e_sample(
        self, dt: Any
    ) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for conditionnestack2e-all data for CodeLLaMA.
        """

        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = dt.etype.split(".")[-1]
        mut_code = dt.mut
        if dt.e_stack_trace:
            mut_code = add_called_comment_to_method(dt.mut, dt.e_stack_trace[0][1])
        etest_name = dt.test_e_key.split("#")[1]
        etest_context = dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        stack_trace_prompt = LLM_stack_trace_prompt(dt, dt.e_stack_trace)
        condition = dt.condition
        if condition and condition != "":
            condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition}\n```\n"
        else:
            condition_prompt = ""
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        paired_data = []
        if not dt.test_ne or len(dt.test_ne) <= 1:
            return paired_data
        else:
            sample_netests = random.sample(list(dt.test_ne), min(5, len(dt.test_ne)))
            sample_size = len(sample_netests)
            ne_test = random.choice(dt.test_ne)
            for _ in range(sample_size):
                if self.config["with_name"]:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                else:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_conditionnestack2e_all(
        self, dt: Any
    ) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for conditionnestack2e-all data for CodeLLaMA.
        """

        mut_name = dt.mut_key.split("#")[1]
        mut_class = dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = dt.etype.split(".")[-1]
        mut_code = dt.mut
        if dt.e_stack_trace:
            mut_code = add_called_comment_to_method(dt.mut, dt.e_stack_trace[0][1])
        etest_name = dt.test_e_key.split("#")[1]
        etest_context = dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        stack_trace_prompt = LLM_stack_trace_prompt(dt, dt.e_stack_trace)
        condition = dt.condition
        if condition and condition != "":
            condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition}\n```\n"
        else:
            condition_prompt = ""
        output = f"""```java\n{dt.test_e}\n```\n</s>"""
        paired_data = []
        if not dt.test_ne:
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Please only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
        else:
            sample_netests = random.sample(list(dt.test_ne), min(5, len(dt.test_ne)))
            for ne_test in sample_netests:
                if self.config["with_name"]:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                else:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_true_nestack2e_all(
        self, nestack2e_dt: Any
    ) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for nestack2e-all data for CodeLLaMA.
        """
        mut_name = nestack2e_dt.mut_key.split("#")[1]
        mut_class = nestack2e_dt.mut_key.split("#")[0].split(".")[-1]
        exception_type = nestack2e_dt.etype.split(".")[-1]

        if nestack2e_dt.e_stack_trace:
            mut_code = add_called_comment_to_method(
                nestack2e_dt.mut, nestack2e_dt.e_stack_trace[0][1]
            )
        etest_name = nestack2e_dt.test_e_key.split("#")[1]
        etest_context = nestack2e_dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        etest_class = nestack2e_dt.test_e_key.split("#")[0].split("/")[-1]
        stack_trace_prompt = LLM_stack_trace_prompt(
            nestack2e_dt, nestack2e_dt.e_stack_trace
        )
        output = f"""```java\n{nestack2e_dt.test_e}\n```\n</s>"""
        paired_data = []
        if not nestack2e_dt.test_ne:
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{nestack2e_dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Please only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{nestack2e_dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
        else:
            sample_netests = random.sample(
                list(nestack2e_dt.test_ne), min(5, len(nestack2e_dt.test_ne))
            )
            for ne_test in sample_netests:
                if self.config["with_name"]:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    instruction = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                    paired_data.append((instruction, output))
                else:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    instruction = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                    paired_data.append((instruction, output))
        return paired_data

    def format_Llama_input_from_true_nestack2e(
        self, nestack2e_dt: Any
    ) -> List[Tuple[str]]:
        """
        Return a tuple of input and output for stackne2e data for CodeLLaMA.
        """
        mut_name = nestack2e_dt.mut_key.split("#")[1]
        mut_class = nestack2e_dt.mut_key.split("#")[0].split(".")[-1]
        exception_type = nestack2e_dt.etype.split(".")[-1]

        if nestack2e_dt.e_stack_trace:
            mut_code = add_called_comment_to_method(
                nestack2e_dt.mut, nestack2e_dt.e_stack_trace[0][1]
            )
        etest_name = nestack2e_dt.test_e_key.split("#")[1]
        etest_context = nestack2e_dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        stack_trace_prompt = LLM_stack_trace_prompt(
            nestack2e_dt, nestack2e_dt.e_stack_trace
        )
        output = f"""```java\n{nestack2e_dt.test_e}\n```\n</s>"""
        paired_data = []
        if not nestack2e_dt.test_ne:
            if self.config["with_name"]:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{nestack2e_dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
            else:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Please only give the new exceptional-behavior test method to complete the following test class'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                output = f"""```java\n{nestack2e_dt.test_e}\n```\n</s>"""
                paired_data.append((input, output))
        else:
            if self.config["with_name"]:
                ne_test = random.choice(nestack2e_dt.test_ne)
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                instruction = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((instruction, output))
            else:
                ne_test = random.choice(nestack2e_dt.test_ne)
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                instruction = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
                paired_data.append((instruction, output))
        return paired_data

    def format_Llama_input_from_stackne2e_all(
        self, stackne2e_dt: Any
    ) -> List[Tuple[str]]:
        mut_name = stackne2e_dt.mut_key.split("#")[1]
        mut_class = stackne2e_dt.mut_key.split("#")[0].split("/")[-1]
        exception_type = stackne2e_dt.etype
        mut_code = stackne2e_dt.mut
        etest_name = stackne2e_dt.test_e_key.split("#")[1]
        etest_context = stackne2e_dt.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        call_stack_prompt = format_stack_trace_prompt(stackne2e_dt)
        output = f"""```java\n{stackne2e_dt.test_e}\n```\n</s>"""

        paired_data = []
        if not stackne2e_dt.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{call_stack_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            output = f"""```java\n{stackne2e_dt.test_e}\n```\n</s>"""
            paired_data.append((input, output))
        for ne_test in stackne2e_dt.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{call_stack_prompt}Here is a related test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            input = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            paired_data.append((input, output))
        return paired_data

    def format_Llama_input_from_covstack2e_no_name(
        self, covstack2e: Any
    ) -> List[Tuple[str]]:
        mut_name = covstack2e.mut_key.split("#")[1]
        mut_class = covstack2e.mut_key.split("#")[0].split("/")[-1]
        exception_type = covstack2e.etype
        mut_code = covstack2e.mut
        etest_name = covstack2e.test_e_key.split("#")[1]
        etest_context = covstack2e.test_context.replace("adhoc_", "")
        call_stack_prompt = format_stack_trace_prompt(covstack2e)
        output = f"""```java\n{covstack2e.test_e}\n```\n</s>"""

        paired_data = []
        if not covstack2e.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{call_stack_prompt}Please only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            instruction = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            output = f"""```java\n{covstack2e.test_e}\n```\n</s>"""
            paired_data.append((instruction, output))
        else:
            ne_test = random.choice(covstack2e.test_ne)
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{call_stack_prompt}Here is a related test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            instruction = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruct}[/INST]"
            paired_data.append((instruction, output))
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


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(DataProcessor, as_positional=False)