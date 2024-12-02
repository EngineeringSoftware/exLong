import code
import collections
import itertools
import random
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import seutil as su
from jsonargparse import CLI
from jsonargparse.typing import Path_drw
from tqdm import tqdm

from etestgen.data.data import TData, DataMUT2E
from etestgen.llm.utils import LLMResults
from etestgen.data.utils import load_dataset
from etestgen.macros import Macros

# from etestgen.model.prediction import Prediction, compute_similarity_metrics

import os


class PredictionInspector:
    def __init__(self):
        self.loaded_data_dir = None
        self.setup = None
        self.eval_set = None
        self.dataset: List[TData] = None
        self.gold_stmts: List[List[str]] = None
        self.gold_insns: List[List[str]] = None
        self.k2preds: Dict[str, List[Prediction]] = {}

        self.loaded_full_data_dir = None
        self.full_dataset: List[TData] = None
        self.indexed_full_dataset: Dict[str, TData] = None

    def load_data(self, setup: str = "mut2e", eval_set: str = "eval/otest"):
        data_dir: Path = Macros.work_dir / "setup" / setup / eval_set
        if self.loaded_data_dir == data_dir:
            # already loaded
            return

        # switching dataset
        print(f"Loading dataset from {data_dir}")
        self.setup = setup
        self.eval_set = eval_set
        with tqdm() as pbar:
            self.dataset = load_dataset(data_dir, clz=DataMUT2E, pbar=pbar)
        # self.gold_stmts = su.io.load(data_dir / "gold_stmts.jsonl")
        print(f"Loaded {len(self.dataset)} data from {data_dir}")
        self.loaded_data_dir = data_dir

        # drop all loaded predictions
        self.k2preds = {}

    def load_full_data(self):
        full_data_dir: Path = Macros.work_dir / "data"
        if self.loaded_full_data_dir == full_data_dir:
            # already loaded
            return

        # loading full dataset
        print(f"Loading full dataset from {full_data_dir}")
        with tqdm() as pbar:
            self.full_dataset = load_dataset(full_data_dir, clz=TData, pbar=pbar)
        self.indexed_full_dataset = {d.id: d for d in self.full_dataset}
        print(f"Loaded {len(self.full_dataset)} data from {full_data_dir}")
        self.loaded_full_data_dir = full_data_dir

    def load_pred(
        self,
        model: str = "chatgpt-basic",
        eval_method: str = "top1",
    ):
        pred_key = f"{model}/{eval_method}"
        if pred_key in self.k2preds:
            # already loaded
            return

        print(f"Loading predictions for {pred_key}")
        pred_dir: Union[Path_drw, Path] = (
            Macros.work_dir / "exp" / self.setup / self.eval_set / pred_key
        )
        self.k2preds[pred_key] = su.io.load(pred_dir / "preds.jsonl", clz=LLMResults)
        print(f"Loaded {len(self.k2preds[pred_key])} preds from {pred_dir}")

    def load_all(
        self,
        setup: str = "CSNm",
        eval_set: str = "eval-any-stmt/val",
        models: List[str] = None,
        eval_methods: List[str] = None,
    ):
        if models is None:
            models = [
                "chatgpt-basic",
                "PipelineModel-ai3-bi3",
                "PipelineModel-as3-bs2",
            ]
        if eval_methods is None:
            eval_methods = ["sampling10", "sampling10-10-sampling1"]
        self.load_full_data()
        self.load_data(setup, eval_set)
        for model, eval_method in itertools.product(models, eval_methods):
            try:
                self.load_pred(model, eval_method)
            except FileNotFoundError:
                print(f"Skipped invalid combination: {model}/{eval_method}")

    def interactive(self):
        code.interact(
            banner="Usage: \n"
            "  inspector.load_data(setup, eval_set)\n"
            "  inspector.load_pred(model, eval_method)\n"
            "  inspector.load_all(setup, eval_set, models, eval_methods)\n"
            "  inspector.view(i)\n"
            "  inspector.compare(model_eval_method_1, model_eval_method_2)\n",
            local={"inspector": self, **locals()},
        )

    def inspect_prompt(self) -> str:
        """
        Inspect the prompts provided to ChatGPT.
        """

        LLM_topk_file = (
            Macros.work_dir
            / "exp"
            / "mut2e"
            / "test"
            / "mut2e-trace-gpt3.5-zero-shot"
            / "top1"
        )
        LLM_topk = load_dataset(save_dir=LLM_topk_file, clz=LLMResults)
        prompt_list = []
        etest_list = []
        data_id = []
        for llm_result in LLM_topk:
            for raw_pred_method in llm_result.topk:
                if "invoke" in raw_pred_method["prompt"][1]["content"]:
                    prompt_list.append(raw_pred_method["prompt"][1]["content"])
                    etest_list.append(llm_result.gold)
                    data_id.append(llm_result.id)
        #
        s = ""

        for i, prompt in enumerate(prompt_list):
            s += f"/** ===== {data_id[i]} ===== **/\n"
            # print(f"// metadata: {data.misc}"s +=
            s += f"// ----- prompt\n"
            s += prompt
            s += "\n"
            s += f"// ----- etest\n"
            s += etest_list[i]
            s += "\n"
            s += "---------------------------\n\n"
        return s

    def inspect_prompt_and_write_to_file(self):
        s = self.inspect_prompt()
        su.io.dump(
            Macros.results_dir / "gpt3.5-trace-prompt-inspection.md",
            s,
            su.io.Fmt.txt,
        )

    def inspect_data_and_write_to_file(
        self, setup: str, eval_set: str, model: str, eval_method: str
    ):
        self.load_data(setup, eval_set)
        self.load_pred(model, eval_method)
        data_inspections = []
        for i in range(len(self.dataset)):
            s = self.view_prediction(i)
            data_inspections.append(s)
        # endfor
        su.io.dump(
            Macros.results_dir
            / f"{setup}-{eval_set}-{model}-{eval_method}-prediction-inspections.java",
            " ".join(data_inspections),
            su.io.Fmt.txt,
        )

    def view_stmt_prediction(
        self,
        data_i: int,
        show_metrics: List[str] = ["code-bleu", "edit-sim"],
        show_after_prediction: bool = True,
    ):
        data = self.dataset[data_i]
        gold_stmt = self.gold_stmts[data_i]
        s = ""
        s += f"/** ===== {data.id} ===== **/\n"
        s += f"// project: {data.project}\n"
        # print(f"// metadata: {data.misc}"s +=
        s += f"// ----- method under test\n"
        s += data.mut
        s += "\n"
        s += f"// ----- test signature\n"
        s += " ".join(data.test_sign.get_tokens())
        s += "\n"
        s += f"// ----- test body"
        s += "\n"

        for stmt_i in range(len(data.test_stmts)):
            s += f"// {stmt_i}: stmt\n"
            s += " ".join(data.test_stmts[stmt_i].get_tokens())
            s += "\n"

        s += "/* vvvvv the statement to predict vvvvv */\n"

        s += "/* ------------------------------------ */\n"
        s += f"// {len(data.test_stmts)}: stmt [GOLD]\n"
        s += " ".join(gold_stmt)
        s += "\n"

        for pred_k, preds in self.k2preds.items():
            s += f"/* ----- {pred_k} */\n"
            pred = preds[data_i]

            for top_i, pred_stmt in enumerate(pred.topk):
                metrics = compute_similarity_metrics(
                    gold_stmt, [pred_stmt["toks"]], k_values=[]
                )
                # s += f"// {top_i}: score=({pred_stmt['score']:.2g}) weight={pred_stmt['weight']}"
                for k in show_metrics:
                    if k not in metrics:
                        continue
                    s += f" {k}={metrics[k]:.2f}"
                    s += "\n"
                s += "/* --- Predicted next statement: --- */\n"
                s += " ".join(pred_stmt["toks"])
                s += "\n\n"
                s += "/* --- Entire predicted test: */\n"
                s += pred.raw_code
                s += "\n"

        s += "/* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */\n"
        return s

    def view_prediction(
        self,
        data_i: int,
        show_metrics: List[str] = ["runnable", "edit-sim"],
        show_after_prediction: bool = True,
    ):
        data = self.dataset[data_i]
        s = ""
        s += f"/** ===== {data.id} ===== **/\n"
        s += f"// project: {data.project}\n"
        # print(f"// metadata: {data.misc}"s +=
        s += f"// ----- method under test\n"
        s += data.mut
        s += "\n"
        s += f"// ----- test body"
        s += "\n"

        s += "/* vvvvv the test to predict vvvvv */\n"

        s += "/* ------------------------------------ */\n"
        s += f"{data.test_e}\n"
        s += "\n"

        # show all conditions predicted by chatgpt

        for pred_k, preds in self.k2preds.items():
            pred: LLMResults = preds[data_i]
            for top_i, prediction in enumerate(pred.topk):
                s += f"/* --- Predicted condition {top_i}: --- */\n"
                s += "// "
                s += prediction["condition"]
                s += "\n"
            s += f"/* ----- {pred_k} */\n"

            for top_i, prediction in enumerate(pred.topk):
                # metrics = compute_similarity_metrics(
                #     gold_stmt, [pred_stmt["toks"]], k_values=[]
                # )
                # s += f"// {top_i}: score=({pred_stmt['score']:.2g}) weight={pred_stmt['weight']}"
                # for k in show_metrics:
                #     if k not in metrics:
                #         continue
                #     s += f" {k}={metrics[k]:.2f}"
                #     s += "\n"
                s += f"/* --- Prompt: --- */\n"
                s += prediction["prompt"]
                s += "\n"

                s += f"/* --- Condition {top_i}: --- */\n"
                s += "// "
                s += prediction["condition"]
                s += "\n"
                s += "/* --- Predicted test: --- */\n"
                s += prediction["prediction"]
                s += "\n\n"
                s += "/* --- Metris: */\n"
                s += "// "
                s += str(prediction["runtime-metrics"])
                s += "\n"

        s += "/* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */\n\n\n"
        return s


if __name__ == "__main__":
    cli = CLI(PredictionInspector, as_positional=False)
