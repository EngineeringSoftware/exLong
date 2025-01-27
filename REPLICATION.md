# Replication Guide

This document is a guide to replicate the experiments for exLong.

## Artifacts
Here are the artifacts required to replicate our experiments and results:
### Dataset:
1. Training
    - [Training dataset](https://utexas.box.com/s/dwxneqvx1m1zw2t68tcugxmk1c7gitjm): The processed dataset (prompts) to train the exLong models.
2. Evaluation
    - [ne2e-test.tar.gz](https://utexas.box.com/shared/static/4m7mntp0ix18dkl1ikkspcmpuvybfs1f.gz): Evaluate the exLong under the **developer-oriented use case**. (data are in directory `rq1-eval/`). The exLong's results are in Table IV and Table V.
    - [machine-view.tar.gz](https://utexas.box.com/shared/static/y4e52k5x8vk8vcr59lg33gebcg2m1caw.gz): Evaluate the exLong under **machine-oriented use case**.(data are in directory `rq2/`). The exLong's results are in Table IX and Figure 4.
    - [netest-diversity.tar.gz](https://utexas.box.com/shared/static/j417e93j1rdvdqz2yobttygfhucfbkjm.gz): Ablation study on how the different nEBTs affect model's performance. (data are in `netest-diversity/`). Results are in Table VII.

### Model checkpoints:
The trained exLong checkpoints which can be directly used for running inference/evaluation.
- [exLong-with-name (7B and 13B)](https://utexas.box.com/s/u20ya44oq8eon8aaot479iynpa90erog): exLong model used in Table IV, Table VI and Table VIII.
- [exLong-no-name (7B)](https://utexas.box.com/s/9oo0fcbnhi8b6tggb273otjt5bzw8u0j): exLong model in Table V.
- [exLong-with-name w.o. stack trace (7B)](https://utexas.box.com/s/qikt46jxnf3g3pvqmznruf9bjd8yi17q): exLong no stack trace model used in Table VI.
- [exLong-with-name w.o. stack trace & guard expr (7B)](https://utexas.box.com/shared/static/mxls9c8580igtbnbt2kw29lxl1nhdsv7.tar): exLong no stack trace & no guard expr model used in Table VI.
- [exLong-with-name w.o. stack trace & guard expr & EBT (7B)](https://utexas.box.com/s/p7bcffw0vxelkrp5a2d70anzkekat2rv): exLong no stack trace & no guard expr & no EBT model used in Table VI.
- [exLong-with-name w.o. stack trace & guard expr & EBT (13B)](https://utexas.box.com/s/uaunxdgzql5m6qqt8113ks288xmk0gh1): exLong 13B no stack trace & no guard expr & no EBT model used in Table VIII.

## Set Up
1. Environment and dependencies

    First make sure you have installed all the required dependencies described in [README.md](./README.md#dependencies-set-up). Note that if you want to train exLong from Code Llama, extra dependencies need to be installed.

2. Experiments setup
    Set up the default directory structure and prepare the dataset as described [here](./README.md#experiments-set-up)

## Experiments
1. Train the exLong models
    (you should use `axolotl` in order for training)
    - with EBT test name in the prompt

    **Note**: 'conditionnestack2e' is the setup name for exLong 
    ```bash
    cd python/
    accelerate launch -m axolotl.cli.train configs/axolotl/axolotl-conditionnestack2e-with-name-7b.yaml
    ```
    You will see checkpoints in directory `_work/exp/conditionnestack2e-with-name-ft/lora-codellama-7b/`

    - Training exLong w.o. EBT name
    ```bash
    cd python/
    accelerate launch -m axolotl.cli.train configs/axolotl/axolotl-conditionnestack2e-no-name-7b.yaml
    ```
    You will see checkpoints in directory `_work/exp/conditionnestack2e-no-name-ft/lora-codellama-7b/`

2. Run inference and evaluation
    
    2.1 Developer-oriented use case
    - Run inference on **developer-oriented use case** (4th row in Table IV, Table V)
    ```bash
    cd python/
    # Run evaluation on the selected 434 examples in the test set
    python -m etestgen.codellama.CodeLLaMA --config_file configs/codellama-7b-conditionnestack2e-with-name-ft.yaml run_gen --split real-test
    ```
    You will see checkpoints, model outputs in directory `_work/exp/conditionnestack2e-with-name-ft/lora-codellama-7b/real-test-set-model-outputs.jsonl`
    - Run evaluation
        - similarity metrics
        ```bash
        python -m etestgen.llm.eval --eval_set test --config_file configs/codellama-7b-conditionnestack2e-with-name-ft.yaml eval_llm_sim
        ```
        - functional correctness metrics
        ```bash
        python -m etestgen.llm.eval --eval_set test --config_file configs/codellama-7b-conditionnestack2e-with-name-ft.yaml eval_runtime_metrics
        ```

    2.2 Machine-oriented Use Case
    - Run inference on **machine-oriented use case** (1st row in Table IX)
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
    - Run evaluation
        - similarity metrics
        ```bash
        python -m etestgen.llm.eval --eval_set test --config_file configs/eval-codellama-7b-machine-view-conditionnestack2e-all-no-name.yaml eval_llm_sim
        ```
        - functional correctness metrics
        ```bash
        python -m etestgen.llm.eval --eval_set test --config_file configs/eval-codellama-7b-machine-view-conditionnestack2e-all-no-name.yaml eval_runtime_metrics
        ```