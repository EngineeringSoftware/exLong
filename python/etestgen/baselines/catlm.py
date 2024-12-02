import random
import torch
from tqdm import tqdm
from typing import Iterable, Any, List, Union, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import seutil as su
from jsonargparse import CLI
from datasets import load_dataset

from etestgen.llm.utils import LLMResults
from etestgen.macros import Macros
from etestgen.llm.utils import postprocess_outputs
from etestgen.codellama.CodeLLaMA import aggregate_predictions_by_id
from etestgen.baselines.DataProcessor import DataProcessor
from etestgen.data.utils import save_dataset


class CATLM:

    def __init__(
        self, config_file: Union[Path, str], random_seed: Optional[int] = None
    ) -> None:
        self.config = su.io.load(config_file)
        self.config_file_path = config_file
        self.split = self.config["split"]
        self.setup = self.config["setup"]
        self.model_name = self.config["model_name"]
        self.exp_dir = Macros.exp_dir / self.config["setup"] / self.model_name
        if random_seed:
            self.exp_dir = (
                Macros.exp_dir
                / self.config["setup"]
                / f"{self.model_name}-{random_seed}"
            )
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
        su.io.mkdir(self.exp_dir, fresh=False)

    def run_gen(self, split: str = "test"):
        self.load_model()
        if split == "test":
            dataset = self.load_test_dataset()
        elif split == "val":
            dataset = self.load_val_dataset()
        elif split == "real-test":
            dataset = self.load_real_test_dataset()
        else:
            raise ValueError(f"Invalid split: {split}")
        model_preds = self.do_inference(dataset)
        su.io.dump(self.exp_dir / f"{self.split}-set-model-outputs.jsonl", model_preds)
        self.format_LLM_predictions(
            self.exp_dir / f"{self.split}-set-model-outputs.jsonl", split=split
        )

    def do_inference(self, dataset: Iterable[Any]) -> List[str]:
        """
        Do inference on the given dataset.
        Cut the prompt into fixed length
        """

        model_preds = []
        with tqdm(total=len(dataset), desc="Inference on the dataset") as pbar:
            for dt in dataset:
                query = dt["instruction"]
                data_id = dt["id"]
                # cut input
                input_ids = self.tokenizer(query, return_tensors="pt").to("cuda")[
                    "input_ids"
                ]
                if self.tokenizer.decode(input_ids[0, -1]) == "</s>":
                    input_ids = input_ids[:, :-1]

                outputs = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    max_new_tokens=512,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.2,
                )
                output_str = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )  # note: the output str includes the input
                model_preds.append((data_id, output_str))
                pbar.update(1)
        return model_preds

    def format_LLM_predictions(self, output_file_name: str, split: str = "test"):
        """
        Format the llama predictions into LLMResults for evaluation.
        """

        if split == "test":
            raw_dataset = DataProcessor(
                config_file=self.config_file_path
            ).load_test_data()
            dataset = self.load_test_dataset()
        # elif split == "val":
        #     raw_dataset = DataProcessor(
        #         config_file=self.config_file_path
        #     ).load_val_data()
        #     dataset = self.load_val_dataset()
        # elif split == "real-test":
        #     raw_dataset = DataProcessor(
        #         config_file=self.config_file_path
        #     ).load_real_test_data()
        #     dataset = self.load_real_test_dataset()
        else:
            raise ValueError(f"Invalid split: {split}")
        pred_dir = Macros.exp_dir / self.config["setup"] / self.model_name
        predictions: List[str] = su.io.load(pred_dir / output_file_name)
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

                    expect_tag = "@Test(expected" + prompt.split("@Test(expected")[1]
                    try:
                        pred_test = expect_tag + pred_str.split(expect_tag)[1]
                    except:
                        pred_test = pred_str
                    # predicted_test = extract_code_from_catlm(mut_dt, pred_test).strip()
                    test = postprocess_outputs(pred_test)
                    topk.append(test)
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

        su.io.mkdir(pred_dir / f"{self.split}-results", fresh=True)
        save_dataset(
            pred_dir / f"{self.split}-results", llm_result_list, clz=LLMResults
        )

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nikitharao/catlm", use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "nikitharao/catlm", device_map="auto"
        )

    def load_test_dataset(self):
        self.test_dataset = load_dataset(
            str(Macros.work_dir / self.config["test_data"]),
            data_files=f"test-{self.setup}.jsonl",
            split="train",
        )
        return self.test_dataset

    def load_real_test_dataset(self):
        self.test_dataset = load_dataset(
            str(Macros.work_dir / self.config["real_test_data"]),
            data_files=self.config["test_data_file"],
            split="train",
        )
        return self.test_dataset

    def load_val_dataset(self):
        self.val_dataset = load_dataset(
            str(Macros.work_dir / self.config["val_data"]),
            data_files=self.config["val_data_file"],
            split="train",
        )
        return self.val_dataset


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


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(CATLM, as_positional=False)
