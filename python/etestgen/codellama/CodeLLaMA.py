from collections import defaultdict
import torch
from datasets import load_dataset
from jsonargparse import CLI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
)
from tqdm import tqdm

import peft
# from trl import SFTTrainer
from pathlib import Path
from typing import *
import seutil as su

from etestgen.macros import Macros
from etestgen.llm.utils import LLMResults, extract_code_from_response
from etestgen.data.utils import save_dataset
from etestgen.codellama.DataProcessor import DataProcessor

logger = su.log.get_logger(__name__, su.log.INFO)


class CodeLlama:

    def __init__(self,
                 config_file: Union[Path, str],
                 train_seed: Optional[int] = None) -> None:
        self.config: Dict[str,Any] = su.io.load(config_file)  # type: ignore
        self.config_file_path = config_file
        self.quant_config = BitsAndBytesConfig(
            **self.config["quant_config"],
            bnb_4bit_compute_dtype=torch.float16)  # do not quant now
        self.base_model_name = self.config["base_model_name"]
        self.model_name = self.config["model_name"]
        if train_seed is not None:
            self.model_name += f"-{train_seed}"
        # self.peft_parameters = LoraConfig(**self.config["lora_config"])
        self.exp_dir = Macros.exp_dir / self.config["setup"] / self.model_name
        self.setup = self.config["setup"]
        self.split = self.config["split"]
        su.io.mkdir(self.exp_dir, fresh=False)

    def load_train_dataset(self):
        """
        Load the dataset for fine-tuning.
        """
        if "training_data" in self.config:
            self.training_dataset = load_dataset(
                str(Macros.work_dir / self.config["training_data"]),
                data_files=self.config["training_data_file"],
                split="train",
            )
        if "val_data" in self.config:
            self.val_dataset = load_dataset(
                str(Macros.work_dir / self.config["val_data"]),
                data_files=self.config["val_data_file"],
                split="train",
            )

    def load_test_dataset(self):
        self.test_dataset = load_dataset(
            str(Macros.work_dir / self.config["test_data"]),
            data_files=self.config["test_data_file"],
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

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name,
                                                       trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # Fix for fp16

    def _load_hf_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
        )
        logger.info("Successfully loaded model")
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

    def load_model(self, target_ckpt: Optional[str] = None):
        if self.config["zero_shot"]:
            self._load_hf_model()
        elif "hf_adapter" in self.config:
            self._load_hf_model()
            lora_config = peft.config.PeftConfig.from_pretrained(
                self.config['hf_adapter']['name'],
                self.config['hf_adapter']['revision'],
            )
            self.model = self.model.add_adapter(lora_config)
        else:
            if target_ckpt is None:
                target_ckpt = self.exp_dir
            logger.info(f"Using {target_ckpt} checkpoint for inference")
            self.model = AutoModelForCausalLM.from_pretrained(
                target_ckpt,
                device_map="auto",
            )
        self.load_tokenizer()

    def load_train_params(self):
        self.train_params = TrainingArguments(
            output_dir=self.exp_dir,
            **self.config["train_args"],
        )

    def formatting_func(self, example: Any):
        prompt_list = []
        for i in range(len(example["input"])):
            query = self.truncate_prompt(example["input"][i],
                                         self.config["prompt_max_length"])
            text = f"{query}\n{example['output'][i]}"
            prompt_list.append(text)
        return prompt_list

    def do_inference(self, dataset: Iterable[Any],
                     generation_config: GenerationConfig) -> List[str]:
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
                query = self.truncate_prompt(query,
                                             self.config["prompt_max_length"])
                inputs = self.tokenizer(query, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=generation_config,
                )
                output_str = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )  # note: the output str includes the input
                model_preds.append((data_id, output_str))
                pbar.update(1)
        return model_preds

    def run_gen(
        self,
        split: str = "test",
        target_ckpt: Optional[str] = None,
        inference_seed: Optional[int] = None,
    ):
        """
        Run the Llama model to do inference. (test set or val set)
        """
        import time
        start_time = time.time()
        if inference_seed:
            torch.manual_seed(inference_seed)
            torch.cuda.manual_seed(inference_seed)
            logger.info(f"Set seed to {inference_seed}")

        self.load_model(target_ckpt)

        if split == "test":
            dataset = self.load_test_dataset()
        elif split == "val":
            dataset = self.load_val_dataset()
        elif split == "real-test":
            dataset = self.load_real_test_dataset()
        else:
            raise ValueError(f"Invalid split: {split}")

        generation_config = GenerationConfig(
            **self.config["GenerationConfig"],
            pad_token_id=self.tokenizer.eos_token_id,
        )
        model_preds = self.do_inference(dataset, generation_config)
        su.io.dump(self.exp_dir / f"{self.split}-set-model-outputs.jsonl",
                   model_preds)
        self.format_llama_predictions(self.exp_dir /
                                      f"{self.split}-set-model-outputs.jsonl",
                                      split=split)
        end_time = time.time()
        print("++++++++++++++++++++++++++++++++++++++")
        print(f"Runtime is {end_time - start_time} seconds")

    ########################
    # helpers #
    ########################

    def truncate_prompt(self, prompt: str, max_length: int) -> str:
        """
        Truncate input to a fixed length
        """

        tokens = self.tokenizer.tokenize(prompt)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            truncated_prompt = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(token_ids))
            truncated_prompt = (
                f"{truncated_prompt}\n```\n [/INST]"  # add the instruction token
            )
            return truncated_prompt
        else:
            return prompt

    def format_llama_predictions(self,
                                 output_file_name: str,
                                 split: str = "test"):
        """
        Format the llama predictions into LLMResults for evaluation.
        """

        if split == "test":
            raw_dataset = DataProcessor(
                config_file=self.config_file_path).load_test_data()
            dataset = self.load_test_dataset()
        elif split == "val":
            raw_dataset = DataProcessor(
                config_file=self.config_file_path).load_val_data()
            dataset = self.load_val_dataset()
        elif split == "real-test":
            raw_dataset = DataProcessor(
                config_file=self.config_file_path).load_real_test_data()
            dataset = self.load_real_test_dataset()
        else:
            raise ValueError(f"Invalid split: {split}")
        pred_dir = Macros.exp_dir / self.config["setup"] / self.model_name
        predictions: List[str] = su.io.load(pred_dir / output_file_name)
        id_2_preds = aggregate_predictions_by_id(predictions, dataset)
        # assert len(id_2_preds) == len(
        #     raw_dataset
        # ), "Number of predictions does not match"
        llm_result_list = []
        with tqdm(total=len(raw_dataset), desc="Processing generated EBTs") as pbar:
            index = 0
            for data_id, pred_list in id_2_preds.items():
                while raw_dataset[index].id != data_id:
                    index += 1
                mut_dt = raw_dataset[index]
                assert mut_dt.id == data_id, "Data id does not match"
                # process topk
                topk = []
                prompts = []
                for prompt, pred_str in pred_list:
                    predicted_test = extract_code_from_response(
                        pred_str.split("[/INST]")[1].strip()).strip()
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

        su.io.mkdir(pred_dir / f"{self.split}-results", fresh=True)
        save_dataset(pred_dir / f"{self.split}-results",
                     llm_result_list,
                     clz=LLMResults)

    @classmethod
    def locate_ckpt(cls, ckpt_dir: Path) -> Optional[Path]:
        ckpt_dirs = list(ckpt_dir.glob("checkpoint-*"))
        if len(ckpt_dirs) == 0:
            ckpt_dirs = None
            logger.info(f"No checkpoint found in {ckpt_dir}")
        elif len(ckpt_dirs) == 1:
            ckpt_file = ckpt_dirs[0]
            logger.info(
                f"Found one checkpoint in {ckpt_dir}: {ckpt_file.name}")
        else:
            ckpt_file = sorted(ckpt_dirs, key=lambda x: x.stat().st_mtime)[-1]
            logger.warning(
                f"Multiple checkpoints found in {ckpt_dir}: {[x.name for x in ckpt_dirs]}; picking the latest modified: {ckpt_file.name}"
            )
        return ckpt_file


def reprocess_raw_dataset(dataset: List, raw_dataset: List):
    new_raw_dataset = []
    for p_dt in dataset:
        data_id = p_dt["id"]
        for dt in raw_dataset:
            if dt.id == data_id:
                new_raw_dataset.append(dt)
                break
    return new_raw_dataset


def aggregate_predictions_by_id(predictions: List, dataset: List):
    """
    Aggregate the predictions w.r.t the prompt by id and return the dict[List]
    """
    id_2_topk = defaultdict(list)
    for pred, dt in zip(predictions, dataset):
        data_id, pred_str = pred
        id_2_topk[data_id].append((dt["instruction"], pred_str))
    return id_2_topk


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(CodeLlama, as_positional=False)
