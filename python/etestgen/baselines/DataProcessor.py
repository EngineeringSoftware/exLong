import seutil as su
from tqdm import tqdm
from typing import *
import random

random.seed(42)
from jsonargparse import CLI
from pathlib import Path

from etestgen.macros import Macros
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.data.data import DataNE2E, parse_data_cls
from etestgen.llm.utils import LLMResults, postprocess_outputs
from etestgen.utils import aggregate_predictions_by_id


class DataProcessor:
    def __init__(self, config_file: Path):
        self.config: dict = su.io.load(config_file)
        self.data_dir = Macros.work_dir / "setup" / self.config["setup"]
        # copy the eval data files to setup dir
        setup_dir = Macros.work_dir / "setup" / self.config["setup"]
        su.io.mkdir(setup_dir / "eval")
        su.bash.run(
            f"cp -r {Macros.data_dir / self.config['source_eval_data']} {setup_dir}/eval/test",
            0,
        )
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
            desc="Processing test data for CAT-LM",
        ) as pbar:
            for data in test_dataset:
                data_list = self.format_data(which=self.config["data_type"])(
                    data, with_name=self.config["with_name"]
                )
                data_id = data.id
                test_dataset_list.extend(
                    [
                        {"id": data_id, "instruction": input, "output": output}
                        for input, output in data_list
                    ]
                )
                pbar.update(1)

        su.io.dump(
            self.data_dir / "eval" / "test" / f"test-{self.setup}.jsonl",
            test_dataset_list,
        )

    def format_LLM_predictions(self, split: str = "test"):
        """
        Format the llama predictions into LLMResults for evaluation.
        """

        raw_dataset = self.load_test_data()
        dataset = su.io.load(
            Macros.work_dir / self.config["test_data"] / f"test-{self.setup}.jsonl"
        )
        pred_dir = Macros.exp_dir / self.config["setup"] / self.config["model_name"]
        predictions = su.io.load(pred_dir / f"{split}-set-model-outputs.jsonl")
        id_2_preds = aggregate_predictions_by_id(predictions, dataset)
        assert len(id_2_preds) == len(
            raw_dataset
        ), "Number of predictions does not match"
        llm_result_list = []
        with tqdm(total=len(raw_dataset), desc="Iterate predictions") as pbar:
            index = 0
            for data_id, pred_list in id_2_preds.items():
                mut_dt = raw_dataset[index]
                assert (
                    f"ne2e-{mut_dt.id}" == data_id or mut_dt.id == data_id
                ), f"Data id does not match {data_id}"
                # process topk
                topk = []
                prompts = []
                for prompt, pred_str in pred_list:
                    num_lines = len(prompt.strip().splitlines())
                    if self.config["with_name"]:
                        generated_text = "\n".join(
                            pred_str.strip().splitlines()[num_lines - 2 :]
                        )
                    else:
                        generated_text = "\n".join(
                            pred_str.strip().splitlines()[num_lines - 1 :]
                        )
                    predicted_test = postprocess_outputs(generated_text)
                    topk.append(predicted_test)
                    prompts.append(prompt)
                gold = raw_dataset[index].test_e
                llm_result = LLMResults(
                    id=data_id,
                    project=mut_dt.project,
                    module_i=mut_dt.module_i,
                    input=mut_dt,
                    mname=mut_dt.mut_key.split("#")[1],
                    cname=mut_dt.mut_key.split("#")[0].split("/")[-1],
                    prompt=prompts,
                    topk=topk,
                    gold=gold,
                )
                llm_result_list.append(llm_result)
                index += 1
                pbar.update(1)
        su.io.mkdir(pred_dir / f"{split}-results", fresh=True)
        save_dataset(pred_dir / f"{split}-results", llm_result_list, clz=LLMResults)

    def format_data(self, which: str):
        if which == "MUT2E":
            return self.format_prompt_for_mut2e
        elif which == "NE2E":
            return self.format_prompt_for_ne2e
        elif which == "ALL":
            return self.format_prompt_for_ne2e_all

    def load_test_data(self):
        data_dir = Macros.work_dir / self.config["test_data"]
        with tqdm("Loading test data") as pbar:
            test_dataset = load_dataset(
                data_dir,
                clz=parse_data_cls(self.config["data_type"]),
                pbar=pbar,
            )
        return test_dataset

    def format_prompt_for_mut2e(self, data: DataNE2E, with_name: bool = False):
        """
        Format the prompt for mut2e setting for CAT-LM.
        """
        mut_class = data.mut_key.split("#")[0].split(".")[-1]
        mut_code = data.mut
        etest_name = data.test_e_key.split("#")[1]
        etype = data.etype.split(".")[-1]
        if with_name:
            etest_context = (
                data.test_context.replace("adhoc_", "")
                .replace(
                    "/*TEST PLACEHOLDER*/",
                    "\n    @Test(expected = "
                    + etype
                    + ".class)\n    public void "
                    + etest_name
                    + "() {\n",
                )
                .strip()
            )
        else:
            etest_context = (
                data.test_context.replace("adhoc_", "")
                .replace(
                    "/*TEST PLACEHOLDER*/",
                    "\n    @Test(expected = " + etype + ".class)\n",
                )
                .strip()
            )
        if etest_context.endswith("}"):
            etest_context = etest_context[:-1]
            etest_context = etest_context.strip()

        prompt = f"""public class {mut_class} {{
    {mut_code}
}}
<|codetestpair|>
{etest_context}
"""
        output = f"""```java\n{data.test_e}\n```\n</s>"""
        return [(prompt, output)]

    def format_prompt_for_ne2e(self, data: DataNE2E, with_name: bool = False):
        """
        Format the prompt for ne2e-random one setting for CAT-LM.
        """
        mut_class = data.mut_key.split("#")[0].split(".")[-1]
        mut_code = data.mut
        etest_name = data.test_e_key.split("#")[1]
        etype = data.etype.split(".")[-1]
        return_data = []
        if data.test_ne:
            ne_test = random.choice(data.test_ne)
            sample_netests = [ne_test]
            for ne_test in sample_netests:
                if with_name:
                    etest_context = (
                        data.test_context.replace("adhoc_", "")
                        .replace(
                            "/*TEST PLACEHOLDER*/",
                            f"\n{ne_test}\n\n    @Test(expected = "
                            + etype
                            + ".class)\n    public void "
                            + etest_name
                            + "() {\n",
                        )
                        .strip()
                    ).strip()
                else:
                    etest_context = (
                        data.test_context.replace("adhoc_", "")
                        .replace(
                            "/*TEST PLACEHOLDER*/",
                            f"\n{ne_test}\n\n    @Test(expected = "
                            + etype
                            + ".class)\n",
                        )
                        .strip()
                    ).strip()
                if etest_context.endswith("}"):
                    etest_context = etest_context[:-1]
                    etest_context = etest_context.strip()
                prompt = f"""public class {mut_class} {{
    {mut_code}
}}
<|codetestpair|>
{etest_context}
"""
                return_data.append((prompt, f"""```java\n{data.test_e}\n```\n</s>"""))
        else:
            if with_name:
                etest_context = (
                    data.test_context.replace("adhoc_", "")
                    .replace(
                        "/*TEST PLACEHOLDER*/",
                        "\n    @Test(expected = "
                        + etype
                        + ".class)\n    public void "
                        + etest_name
                        + "() {\n",
                    )
                    .strip()
                )
            else:
                etest_context = (
                    data.test_context.replace("adhoc_", "")
                    .replace(
                        "/*TEST PLACEHOLDER*/",
                        "\n    @Test(expected = " + etype + ".class)\n",
                    )
                    .strip()
                )
            if etest_context.endswith("}"):
                etest_context = etest_context[:-1]
                etest_context = etest_context.strip()
            prompt = f"""
public class {mut_class} {{
    {mut_code}
}}
<|codetestpair|>
{etest_context}
"""
            return_data.append((prompt, f"""```java\n{data.test_e}\n```\n</s>"""))

        return return_data

    def format_prompt_for_ne2e_all(self, data: DataNE2E, with_name: bool = False):
        """
        Format the prompt for mut2e setting for CAT-LM.
        """
        mut_class = data.mut_key.split("#")[0].split(".")[-1]
        mut_code = data.mut
        etest_name = data.test_e_key.split("#")[1]
        etype = data.etype.split(".")[-1]
        return_data = []
        if data.test_ne:
            sample_netests = random.sample(
                list(data.test_ne), min(5, len(data.test_ne))
            )
            for ne_test in sample_netests:
                if with_name:
                    etest_context = (
                        data.test_context.replace("adhoc_", "")
                        .replace(
                            "/*TEST PLACEHOLDER*/",
                            f"\n{ne_test}\n\n    @Test(expected = "
                            + etype
                            + ".class)\n    public void "
                            + etest_name
                            + "() {\n",
                        )
                        .strip()
                    ).strip()
                else:
                    etest_context = (
                        data.test_context.replace("adhoc_", "")
                        .replace(
                            "/*TEST PLACEHOLDER*/",
                            f"\n{ne_test}\n\n    @Test(expected = "
                            + etype
                            + ".class)\n",
                        )
                        .strip()
                    ).strip()
                if etest_context.endswith("}"):
                    etest_context = etest_context[:-1]
                    etest_context = etest_context.strip()
                prompt = f"""public class {mut_class} {{
    {mut_code}
}}
<|codetestpair|>
{etest_context}
"""
                return_data.append((prompt, f"""```java\n{data.test_e}\n```\n</s>"""))
        else:
            if with_name:
                etest_context = (
                    data.test_context.replace("adhoc_", "")
                    .replace(
                        "/*TEST PLACEHOLDER*/",
                        "\n    @Test(expected = "
                        + etype
                        + ".class)\n    public void "
                        + etest_name
                        + "() {\n",
                    )
                    .strip()
                )
            else:
                etest_context = (
                    data.test_context.replace("adhoc_", "")
                    .replace(
                        "/*TEST PLACEHOLDER*/",
                        "\n    @Test(expected = " + etype + ".class)\n",
                    )
                    .strip()
                )
            if etest_context.endswith("}"):
                etest_context = etest_context[:-1]
                etest_context = etest_context.strip()
            prompt = f"""
public class {mut_class} {{
    {mut_code}
}}
<|codetestpair|>
{etest_context}
"""
            return_data.append((prompt, f"""```java\n{data.test_e}\n```\n</s>"""))

        return return_data


def extract_code_from_catlm(data: Any, gen_code: str):
    etest_class = data.test_e_key.split("#")[0].split("/")[-1]
    code_lines = gen_code.splitlines()
    class_start_line = 0
    for i in range(len(code_lines)):
        if "public class" in code_lines[i]:
            class_start_line = i
            break
    gen_code = "\n".join(code_lines[class_start_line:])
    return gen_code


def remove_test_file_header(gen_code: str):
    code_lines = gen_code.splitlines()
    class_start_line = 0
    for i in range(len(code_lines)):
        if "public class" in code_lines[i]:
            class_start_line = i
            break
    gen_code = "\n".join(code_lines[class_start_line + 1 :])
    gen_code = gen_code.replace("\n\n\n", "")
    return gen_code


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(DataProcessor, as_positional=False)
