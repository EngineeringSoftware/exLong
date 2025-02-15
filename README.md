# 🐲🔨 exLong: Generating Exceptional Behavior Tests with Large Language Models 
exLong is a large language model instruction-tuned from CodeLlama and embeds reasoning about 
- **traces** that lead to throw statements
- **conditional expressions** that guard throw statements
- **non-exceptional behavior tests** that execute similar traces

# About
This repo hosts the code and data for the following ICSE 2025 paper:

Title: [exLong: Generating Exceptional Behavior Tests with Large Language Models](https://arxiv.org/abs/2405.14619)

Authors: [Jiyang Zhang](https://jiyangzhang.github.io/), [Yu Liu](https://sweetstreet.github.io/), [Pengyu Nie](https://pengyunie.github.io/), [Junyi Jessy Li](https://jessyli.com/), [Milos Gligoric](http://users.ece.utexas.edu/~gligoric/)



```bibtex
@inproceedings{ZhangETAL25exLong,
  author = {Zhang, Jiyang and Liu, Yu and Nie, Pengyu and Li, Junyi Jessy and Gligoric, Milos},
  title = {exLong: Generating Exceptional Behavior Tests with Large Language Models},
  booktitle = {International Conference on Software Engineering},
  year = {2025},
}
```

# Table of Contents
1. [Quick Start][sec-hf] 🤗
2. [Set Up][sec-setup] :rocket:
3. [Experiments][sec-exp] :construction_worker:
4. [Artifacts][sec-artifacts] :star:


# Quick Start
 
[sec-hf]: #quick-start

- The exLong dataset is on Hugging Face [🤗](https://huggingface.co/datasets/EngineeringSoftware/exLong-dataset)!
```bash
from datasets import load_dataset

with_name_ds = load_dataset("EngineeringSoftware/exLong-dataset", "with-EBT-name")
no_name_ds = load_dataset("EngineeringSoftware/exLong-dataset", "no-EBT-name")
```

- The exLong model is on Hugging Face [🤗](https://huggingface.co/EngineeringSoftware/exLong)!

```bash
pip install transformers accelerate bitsandbytes peft
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the base model
base_model_name = "codellama/CodeLlama-7b-Instruct-hf"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the LoRA configuration
peft_model_id = "EngineeringSoftware/exLong"
config = PeftConfig.from_pretrained(peft_model_id, revision="with-etest-name")  # set revision to "no-etest-name" for no EBT name

# Load the LoRA model
model = PeftModel.from_pretrained(base_model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

prompt = """<s>[INST] <<SYS>>
You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code.
<</SYS>>

Please complete an exceptional behavior test method in Java to test the method 'factorial' for the exception 'IllegalArgumentException'.
The method to be tested is defined as:
```java
public static long factorial(int n) {
    if (n < 0) {
        throw new IllegalArgumentException("Number must be non-negative.");
    }
    long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}
` ` `
Please only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:
```java
public class FactorialTest {
}
` ` `
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate code
output = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.2,      # Sampling temperature (lower is more deterministic)
    top_p=0.95,           # Top-p (nucleus) sampling
    do_sample=True        # Enable sampling
)

# Decode and print the generated code
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Code:")
print(generated_code)
```

# Set Up
[sec-setup]: #set-up

## Dependencies Set Up
[sec-dep]: #dependencies-set-up

1. Create conda environment
```bash
conda create -n exlong python=3.9
conda activate exlong
pip install -r requirements.txt
```

2. We used [axolotl](https://github.com/axolotl-ai-cloud/axolotl) to fine-tune the CodeLlama model. If you want to train your own model, install the extra dependencies
```bash
# we used an older version of axolotl to train the models
git clone git@github.com:JiyangZhang/axolotl-exlong.git
cd axolotl-exlong/
conda activate exlong
pip install packaging
# set CUDA_HOME
export CUDA_HOME=/opt/apps/cuda/12.0/
pip3 install -e '.[flash-attn,deepspeed]'
```
## Experiments Set Up
[sec-setupexp]: #experiments-set-up

1. Download raw dataset
```bash
mkdir -p _work/data/
mkdir -p _work/exp/
mkdir -p _work/setup/

wget -L https://utexas.box.com/shared/static/hfcp4za3j9vp8lh5u8iviadixuxu8080.gz -O raw-data.tar.gz
tar -xzf raw-data.tar.gz -C _work/data/
mv _work/data/etestgen-raw-data-12k _work/data/ne2e

wget -L https://utexas.box.com/shared/static/4m7mntp0ix18dkl1ikkspcmpuvybfs1f.gz -O ne2e-test.tar.gz
tar -xzf ne2e-test.tar.gz -C _work/data/

wget -L https://utexas.box.com/shared/static/y4e52k5x8vk8vcr59lg33gebcg2m1caw.gz -O rq2.tar.gz
tar -xzf rq2.tar.gz -C _work/data/

# netest-diversity
wget -L https://utexas.box.com/shared/static/j417e93j1rdvdqz2yobttygfhucfbkjm.gz -O netest-diversity.tar.gz
tar -xzf netest-diversity.tar.gz -C _work/data/

```
You should see  `_work/data/ne2e`, `_work/data/rq1-eval`, `_work/data/rq2` and `_work/data/netest-diversity`.

2. Prepare dataset and put them in the `_work/setup` directory
-  exLong && exlong sample (Table IV & V)
```bash
# exlong
inv -e data.setup-model-data --setup-name conditionnestack2e-with-name-ft
inv -e data.setup-model-data --setup-name conditionnestack2e-no-name-ft

# exlong sample
inv -e data.setup-model-data --setup-name conditionnestack2e-all-with-name-ft
inv -e data.setup-model-data --setup-name conditionnestack2e-all-no-name-ft
```
You should see `_work/setup/conditionnestack2e-with-name-ft/`, `_work/setup/conditionnestack2e-no-name-ft/`, `_work/setup/conditionnestack2e-all-with-name-ft/`, `_work/setup/conditionnestack2e-all-no-name-ft/` directories.


3. Construct prompts for exLong developer-view
- exLong
```bash
inv -e data.process-codellama-data --setup-name conditionnestack2e-with-name-ft
inv -e data.process-codellama-data --setup-name conditionnestack2e-no-name-ft
```

4. Construct prompts for exLong machine-view
```bash
mkdir _work/setup/conditionnestack2e-all-no-name-ft/eval/ -p
cp -r _work/data/rq2/ _work/setup/conditionnestack2e-all-no-name-ft/eval/
python -m etestgen.codellama.realDataProcessor --config_file configs/eval-codellama-7b-machine-view-conditionnestack2e-all-no-name.yaml process_test_data
```
You will see `_work/setup/conditionnestack2e-all-no-name-ft/eval/rq2/test-conditionnestack2e-all-no-name-ft.jsonl`.

# Experiments

[sec-exp]: #experiments

## Training

1. Training exLong w. EBT name

**Note**: conditionnestack2e is the setup name for exLong 
```bash
cd python/
accelerate launch -m axolotl.cli.train configs/axolotl/axolotl-conditionnestack2e-with-name-7b.yaml
```
You will see checkpoints in directory `_work/exp/conditionnestack2e-with-name-ft/lora-codellama-7b/`

2. Training exLong w.o. EBT name
```bash
cd python/
accelerate launch -m axolotl.cli.train configs/axolotl/axolotl-conditionnestack2e-no-name-7b.yaml
# script to run on TACC
sbatch axolotl-lora-codellama-7b-conditionnestack2e-no-name.sh
```
You will see checkpoints in directory `_work/exp/conditionnestack2e-no-name-ft/lora-codellama-7b/`

3. Running inference exLong for developer-view
```bash
cd python/
# Run evaluation on the selected 434 examples in the test set
python -m etestgen.codellama.CodeLLaMA --config_file configs/codellama-7b-conditionnestack2e-with-name-ft.yaml run_gen --split real-test
```
You will see checkpoints, model outputs in directory `_work/exp/conditionnestack2e-with-name-ft/lora-codellama-7b/real-test-set-model-outputs.jsonl`

4. Running inference exLong for machine-view
```bash
cd python/
python -m etestgen.codellama.CodeLLaMA --config_file configs/eval-codellama-7b-machine-view-conditionnestack2e-all-no-name.yaml run_gen
# Evaluation1: all covered projects
python -m etestgen.llm.eval --config_file configs/eval-codellama-7b-machine-view-conditionnestack2e-all-no-name.yaml eval_runtime_metrics
# You will see eval results in `results/model-results/conditionnestack2e-all-no-name-ft-lora-codellama-7b-eval-rq2-runtime-metrics.json`
# Evaluation2: intersection projects
python -m etestgen.llm.eval --eval_set rq2 --config_file configs/eval-codellama-7b-machine-view-conditionnestack2e-all-no-name.yaml eval_subset_llm_results --subset_id_file ../results/tool-results/intersect-ids.json
# You will see eval results in `results/model-results/conditionnestack2e-all-no-name-ft-lora-codellama-7b-eval-rq2-intersect-runtime-metrics.json`
```
You will see model generations in directory `_work/exp/conditionnestack2e-all-no-name-ft/lora-codellama-7b/rq2-model-outputs.jsonl`


## Evaluation: compute metrics

Given test cases generated by exLong, this step will evaluate them with metrics like BLEU, CodeBLEU, Test Coverage, etc.

- Input/Output info
    - Dataset used to evaluation is expected at `_work/{test_data}`
    - Processed LLM prediction is expected at `_work/exp/{setup}/{model_name}/test-results`
    - Similarity metrics result will be written to `_work/exp/{setup}/{model_name}/test-out/similarity_metrics_summary.json` and `results/model-results/{setup}-{exp}-{eval_set}-sim-metrics.json`
    - Runtime Metrics will be written to `results/model-results/{setup}-{exp}-{eval_set}-runtime-metrics.json` and individual result will be at `_work/exp/{setup}/{model_name}/test-results/metrics.jsonl`


- To run evaluation on similarity metrics
  - Run on an individual experiment

    ```bash
    python -m etestgen.llm.eval --eval_set test --config_file [/path/to/config/file] eval_llm_sim
    ```

- To run evaluation on runtime metrics
  - Run on an individual experiment

    ```bash
    python -m etestgen.llm.eval --eval_set test --config_file [/path/to/config/file] eval_runtime_metrics
    ```

## Ablations on exLong's context
#### Diversity of the nEBTs
1. Prepare Dataset
```bash
mkdir -p _work/setup/diversity-conditionnestack2e-sample-with-name-ft/real-eval/test/
mkdir -p _work/setup/diversity-conditionnestack2e-all-with-name-ft/real-eval/test/
cp -r _work/data/netest-diversity/* _work/setup/diversity-conditionnestack2e-sample-with-name-ft/real-eval/test/
cp -r _work/data/netest-diversity/* _work/setup/diversity-conditionnestack2e-all-with-name-ft/real-eval/test/
cd python/
python -m etestgen.codellama.DataProcessor --config_file configs/codellama-7b-diversity-conditionnestack2e-sample-with-name-ft.yaml process_real_test_data
python -m etestgen.codellama.DataProcessor --config_file configs/codellama-7b-diversity-conditionnestack2e-all-with-name-ft.yaml process_real_test_data
```
You will see processed data in `_work/setup/diversity-conditionnestack2e-all-with-name-ft/real-eval/test/` and `_work/setup/diversity-conditionnestack2e-sample-with-name-ft/real-eval/test/`.

2. Running Inference
```bash
# [2nd row] use the same exLong ckpt but try prompting with the same nEBT multiple times
python -m etestgen.codellama.CodeLLaMA --config_file configs/codellama-7b-diversity-conditionnestack2e-sample-with-name-ft.yaml  run_gen --split real-test --target_ckpt ../_work/exp/conditionnestack2e-with-name-ft/lora-codellama-7b/
# [3rd row] use the same exLong ckpt but try prompting with different nEBTs
python -m etestgen.codellama.CodeLLaMA --config_file configs/codellama-7b-diversity-conditionnestack2e-all-with-name-ft.yaml  run_gen --split real-test --target_ckpt ../_work/exp/conditionnestack2e-with-name-ft/lora-codellama-7b/
```
You will see model outputs in directory `_work/exp/diversity-conditionnestack2e-sample-with-name-ft/lora-codellama-7b/` and `_work/exp/diversity-conditionnestack2e-all-with-name-ft/lora-codellama-7b/`

#### exLong w.o. stack trace

1. Prepare Dataset
```bash
inv -e data.setup-model-data --setup-name conditionne2e-with-name-ft
inv -e data.process-codellama-data --setup-name conditionne2e-with-name-ft
```
2. Running Inference
```bash
python -m etestgen.codellama.CodeLLaMA --config_file python/configs/eval/codellama-7b-conditionne2e-with-name-ft.yaml run_gen
```

#### exLong w.o. stack trace & guard expression

1. Prepare Dataset
```bash
inv -e data.setup-model-data --setup-name ne2e-with-name-ft
inv -e data.process-codellama-data --setup-name ne2e-with-name-ft
```

2. Running Inference
```bash
python -m etestgen.codellama.CodeLLaMA --config_file configs/eval/codellama-7b-ne2e-with-name-ft.yaml run_gen
```

#### exLong w.o. stack trace & guard expression & nEBT

1. Prepare Dataset
```bash
inv -e data.setup-model-data --setup-name mut2e-with-name-ft
inv -e data.process-codellama-data --setup-name mut2e-with-name-ft
```

2. Running Inference
```bash
python -m etestgen.codellama.CodeLLaMA --config_file configs/eval/codellama-7b-mut2e-with-name-ft.yaml run_gen
```


# Artifacts:

[sec-artifacts]: #artifacts

### Model Checkpoints:
- [exLong-with-name (7B and 13B)](https://utexas.box.com/s/u20ya44oq8eon8aaot479iynpa90erog): exLong models in Table IV, Table VI and Table VIII.
- [exLong-no-name (7B)](https://utexas.box.com/s/9oo0fcbnhi8b6tggb273otjt5bzw8u0j): exLong models in Table V.
- [exLong-with-name w.o. stack trace (7B)](https://utexas.box.com/s/qikt46jxnf3g3pvqmznruf9bjd8yi17q): exLong no stack trace model in Table VI.
- [exLong-with-name w.o. stack trace & guard expr (7B)](https://utexas.box.com/shared/static/mxls9c8580igtbnbt2kw29lxl1nhdsv7.tar): exLong no stack trace & no guard expr model in Table VI.
- [exLong-with-name w.o. stack trace & guard expr & EBT (7B)](https://utexas.box.com/s/p7bcffw0vxelkrp5a2d70anzkekat2rv): exLong no stack trace & no guard expr & no EBT model in Table VI.
- [exLong-with-name w.o. stack trace & guard expr & EBT (13B)](https://utexas.box.com/s/uaunxdgzql5m6qqt8113ks288xmk0gh1): exLong 13B no stack trace & no guard expr & no EBT model in Table VIII.

### Dataset:
- [repos.tar.gz](https://utexas.box.com/s/5f9ogvbe3nnz2fijplu1zs2ohygmgsv2): The repository list from which we collected the dataset.
- [raw-data.tar.gz](https://utexas.box.com/shared/static/hfcp4za3j9vp8lh5u8iviadixuxu8080.gz): The raw collected data from the open-source repositories. `etestgen-raw-data-12k/`
- [ne2e-test.tar.gz](https://utexas.box.com/shared/static/4m7mntp0ix18dkl1ikkspcmpuvybfs1f.gz): The collected dataset for eval in **developer-view**. `rq1-eval/`
- [machine-view.tar.gz](https://utexas.box.com/shared/static/y4e52k5x8vk8vcr59lg33gebcg2m1caw.gz): The collected dataset for eval in **machine-view**. `rq2/`
- [netest-diversity.tar.gz](https://utexas.box.com/shared/static/j417e93j1rdvdqz2yobttygfhucfbkjm.gz): The collected dataset we use to study how the different nEBTs affect model's performance (Table VII). `netest-diversity/`
- [processed dataset](https://utexas.box.com/s/dwxneqvx1m1zw2t68tcugxmk1c7gitjm): The processed dataset (prompts) to train the exLong models.
