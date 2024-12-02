from collections import defaultdict
from typing import *
from jsonargparse import CLI

# from chatgpt_wrapper import ChatGPT
from jsonargparse.typing import Path_dc, Path_drw
import seutil as su
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from etestgen.macros import Macros
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.data.data import parse_data_cls
from etestgen.llm.utils import (
    LLMResults,
    LLMOutput,
    LLMConfig,
    extract_code_from_response,
)

logger = su.log.get_logger(__name__)
# openai.organization = "org-yF1uZ50476QV6DGIWNxAdov4"  # Milos's
# openai.organization = "org-ccIyHj49SsrKUQWXHrIRc8tD"
# openai.api_key = "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"  # Milos's
# openai.api_key = "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"  # Jiyang


class OpenAIRunner:
    def __init__(self, config_file: str):
        self.config = su.io.load(config_file)
        self.model_name = self.config["model_name"]
        self.data_dir = (
            Macros.work_dir / "setup" / self.config["setup"] / self.config["model_name"]
        )
        self.exp_dir = Macros.exp_dir / self.config["setup"] / self.config["model_name"]
        su.io.mkdir(self.exp_dir)

    def ask_gpt_model(self, selected_ids: List[Any] = None):
        # prepare
        gpt_outputs = defaultdict(list)
        # load prompts
        prompts_data: List[Dict] = su.io.load(self.data_dir / "prompt.jsonl")
        if selected_ids is not None:
            prompts_data = [p for p in prompts_data if p["id"] in selected_ids]
        with tqdm(prompts_data, desc=f"Asking {self.model_name}...") as pbar:
            for pt in prompts_data:
                output = self.ask_chatgpt_for_code(pt["messages"])
                gpt_outputs[pt["id"]].append(output)
                pbar.update(1)
        # write results
        su.io.dump(self.exp_dir / "model-outputs.json", gpt_outputs)
        self.extract_gpt_outputs()

    def extract_gpt_outputs(self):
        # prepare
        llm_output_dir = self.exp_dir / "test-results"
        su.io.mkdir(llm_output_dir)
        # load raw dataset
        raw_dataset = load_dataset(
            Macros.data_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        gpt_outputs = su.io.load(self.exp_dir / "model-outputs.json")
        llm_results = []
        for data_id, gpt_output in gpt_outputs.items():
            for dt in raw_dataset:
                if dt.id == data_id:
                    topk_preds = []
                    for m_out in gpt_output:
                        for topk_res in m_out["responses"]:
                            predicted_test = extract_code_from_response(
                                topk_res
                            ).strip()
                            topk_preds.append(predicted_test)

                    cname = dt.test_e_key.split("#")[0].replace("/", ".")
                    mname = dt.test_e_key.split("#")[1]
                    etest = dt.test_e
                    pred = LLMResults(
                        id=dt.id,
                        project=dt.project,
                        input=dt,
                        cname=cname,
                        mname=mname,
                        module_i=dt.module_i,
                        prior_stmts=None,
                        topk=topk_preds,
                        prompt=[g["prompt"] for g in gpt_output],
                        gold=etest,
                    )
                    llm_results.append(pred)

                    break
        save_dataset(llm_output_dir, llm_results)

    ####
    # Helper functions
    ####
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(20))
    def ask_chatgpt_for_code(self, prompt: List[Dict]) -> LLMOutput:
        """
        Ask Chat-based LLM to write code.

        Note: it is expected to accept the chat-based prompt (i.e. with system and user messages)

        Args:
        - prompt [List[Dict]]: the prompt to ask Chat-based LLM to write code

        """

        sample_size = self.config["sample_size"]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=prompt,
            temperature=self.config["temperature"],
            n=sample_size,
            max_tokens=256,
            stop=["```\n"],
        )
        responses = []
        for i in range(sample_size):
            responses.append(response["choices"][i]["message"]["content"])
        output = LLMOutput(responses, prompt)
        return output


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(OpenAIRunner, as_positional=False)
