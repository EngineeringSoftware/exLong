import re
import time
import seutil as su
import ollama
import json
from torch.cuda import temperature
from tqdm import tqdm

from transformers import GenerationConfig
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterable, Union
from collections import namedtuple

from etestgen.codellama.CodeLLaMA import CodeLlama
from etestgen.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)


class QuantLlama(CodeLlama):

    def __init__(self,
                 config_file: Union[Path, str],
                 train_seed: Optional[int] = None) -> None:
        self.config: Dict[str, Any] = su.io.load(config_file)  # type: ignore
        self.config_file_path = config_file
        self.base_model_name = self.config["base_model_name"]
        self.model_name = self.config["model_name"]
        if train_seed is not None:
            self.model_name += f"-{train_seed}"
        # self.peft_parameters = LoraConfig(**self.config["lora_config"])
        self.exp_dir = Macros.exp_dir / self.config["setup"] / self.model_name
        self.setup = self.config["setup"]
        self.split = self.config["split"]
        self.load_tokenizer()
        su.io.mkdir(self.exp_dir, fresh=False)

    @staticmethod
    def setup_model(
        model_type: Optional[str] = None,
        target_ckpt: Optional[str] = None,
    ):
        if target_ckpt is not None:
            model_file = f"""
            FROM codellama:7b-python-q6_K
            ADAPTER {target_ckpt}
            """

            for response in ollama.create(
                    f"linghanz/exlong:{model_type}",
                    modelfile=model_file,
                    stream=True,
            ):
                print(response['status'])

        elif model_type is not None:
            ollama.pull(f"linghanz/exLong:{model_type}")
        else:
            raise ValueError("please give model type or target ckpt")

    def load_model(self, target_ckpt: Optional[str] = None):
        if target_ckpt is not None:
            self.setup_model(target_ckpt=target_ckpt)
        else:
            all_models = [model.model for model in ollama.list().models]
            if f"exlong:{self.config['model_type']}" not in all_models:
                self.setup_model(self.config['model_type'])
        self.model = f"linghanz/exlong:{self.config['model_type']}"

    def do_inference(self, dataset: Iterable[Any],
                     generation_config: GenerationConfig) -> List[str]:
        """
        Do inference on the given dataset.
        Cut the prompt into fixed length
        """

        model_preds = []
        seen_ids = set()
        all_ids = set([dt['id'] for dt in dataset])
        with tqdm(
                total=len(all_ids),
                desc="Generating tests for each throw statement",
        ) as pbar:
            with open( self.exp_dir / "time.json" , "w") as time_file:
                for dt in dataset:
                    start_time = time.time()
                    query = dt["instruction"]
                    data_id = dt["id"]
                    if data_id not in seen_ids:
                        if len(seen_ids) != 0:
                            pbar.update(1)
                        seen_ids.add(data_id)
                    # cut input
                    query = self.truncate_prompt(
                        query,
                        self.config["prompt_max_length"],
                    )
                    chat_match = re.search(r"<<SYS>>([\S\s]*)<<\/SYS>>([\S\s]*)\[\/INST\]", query, flags=re.MULTILINE)
                    # print(query)
                    assert chat_match is not None

                    chat = [{"role": "system", "content": chat_match.group(1)}, {"role": "user", "content": chat_match.group(2)}]
                    model_output = ollama.chat(
                        model=self.model,
                        messages=chat,
                        options={
                            "temperature": generation_config.temperature,
                            "num_predict": generation_config.max_new_tokens,
                        })
                    output_str = model_output["message"]['content']
                    if "[/INST]" not in output_str:
                        output_str = "[/INST]" + output_str
                    model_preds.append((data_id, output_str))
                    time_file.write(json.dumps({"id": dt['id'], "time": time.time() - start_time}))
                    pbar.update(1)
            print(f"Total {len(model_preds)} test generated for {len(all_ids)} throw statements")
        return model_preds
