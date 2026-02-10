import copy
import datetime as dt
import json
import os
import sys
from argparse import ArgumentParser

import datasets
from common.cli_constants import LM_E2E_MODELS, NN_E2E_MODELS
from common import global_vars
from common.args_to_config import (get_grep_mtc_experiment_configs,
                                   get_lm_e2e_experiment_configs,
                                   get_nn_e2e_experiment_configs,
                                   get_sim_mtc_experiment_configs,
                                   get_terms_mtc_experiment_configs)
from common.cli_args import add_base_args, add_config_args
from common.constants import (DATA_DIRPATH, EXPERIMENT_SESSION_DIRPATH, LABEL_COL,
                              RANDOM_SEED, RAW_FILENAME, TEXT_1_COL,
                              TEXT_2_COL, TEXT_COL, UNSLOTH_TRAINING_KEY,
                              VUL4J_TEST_FILEPATH, CommonConfigKeys,
                              End2EndName)
from common.load_dataset import (load_vul4j_for_e2e, load_vul4j_for_fnd,
                                 load_vul4j_for_lnk)
from common.utils_training import NpEncoder, print_stdout_file, split_dataset
from evaluate import logging as el
from evaluation.evaluate_common import config_to_str
from evaluation.evaluate_fix import eval_fix
from evaluation.evaluate_grep_mtc import eval_grep_mtc
from evaluation.evaluate_lm import evaluate_lm_e2e
from evaluation.evaluate_nn import evaluate_nn_e2e
from evaluation.evaluate_sim_mtc import eval_sim_mtc
from evaluation.evaluate_terms_mtc import eval_terms_mtc
from pandas import DataFrame

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not sys.stdout.isatty():
        datasets.disable_progress_bar()
        el.disable_progress_bar()

    argparser = ArgumentParser()
    argparser = add_base_args(argparser, "end-to-end", "input", "output", "debug", "cache")
    argparser = add_config_args(argparser, "splits", "cv", "metric", "loss")
    argparser = add_config_args(argparser, "archi-style", "train-type", "ft-epochs", "ft-augment", "sus-threshold")
    argparser = add_config_args(argparser, "unsloth-training-both", "unsloth-training")
    argparser = add_config_args(argparser, "fnd-epochs", "fnd-hidden-size-1", "fnd-hidden-size-2", "fnd-augment", "fnd-loss", "fnd-one-line", "fnd-one-line-both")
    argparser = add_config_args(argparser, "lnk-epochs", "lnk-hidden-size-1", "lnk-hidden-size-2", "lnk-augment", "lnk-loss", "lnk-one-line", "lnk-one-line-both")
    argparser = add_config_args(argparser, "use-cwe", "use-cwe-both", "merge")
    argparser = add_config_args(argparser, "separate-both", "separate-yes", "grep-matches", "grep-extended-both", "grep-extended")
    argparser = add_config_args(argparser, "terms-threshold", "sim-extractor", "sim-threshold", "keywords-nr", "keywords-dedup")
    args = argparser.parse_args()

    model_id = args.end_to_end

    default_outdir = os.path.join(EXPERIMENT_SESSION_DIRPATH, model_id)
    if args.output is None:
        print(f"Directory where to write the output results not supplied. Using the default location {default_outdir}.")
        out_dirpath = os.path.abspath(default_outdir)
    else:
        out_dirpath = os.path.abspath(args.output)
    session_outdir = os.path.join(out_dirpath, dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(session_outdir, exist_ok=True)
    session_outfile = os.path.join(session_outdir, RAW_FILENAME)
    with open(session_outfile, "w") as _:
        pass
    if args.input is None:
        print_stdout_file(f"Directory containing the Vul4J dataset not supplied. Using the default expected location in \"{DATA_DIRPATH}\"", session_outfile)
        dataset_basepath = os.path.abspath(DATA_DIRPATH)
    else:
        dataset_basepath = os.path.abspath(args.input)

    print_stdout_file("Loading Vul4J Dataset for End-to-End model...", session_outfile)
    e2e_vul4j_df = load_vul4j_for_e2e(VUL4J_TEST_FILEPATH, dataset_basepath)
    fnd_vul4j_df = load_vul4j_for_fnd(VUL4J_TEST_FILEPATH, dataset_basepath)
    lnk_vul4j_df = load_vul4j_for_lnk(VUL4J_TEST_FILEPATH, dataset_basepath)
    if any(e is None for e in [e2e_vul4j_df, fnd_vul4j_df, lnk_vul4j_df]):
        print_stdout_file("Failed to load Vul4J dataset. Exiting", session_outfile)
        exit(1)
    print_stdout_file("Distribution of the labels in E2E dataset", session_outfile)
    print_stdout_file(e2e_vul4j_df[LABEL_COL].value_counts(), session_outfile)

    print_stdout_file(f"[{dt.datetime.now()}] Starting training-validation-testing session of {model_id}", session_outfile)
    if args.splits is None:
        print_stdout_file("No dataset splits specified. Using default [0.7-0.15-0.15].", session_outfile)
        splits = [[0.7, 0.15, 0.15]]
    else:
        splits = [[float(v) for v in split.split("-")] for split in args.splits.split(",")]
    if args.cv is None:
        print_stdout_file("No Monte Carlo Cross-Validation specified. Doing just one round.", session_outfile)
        cv = 1
    else:
        cv = int(args.cv)
        print_stdout_file(f"Running Monte Carlo Cross-Validation for {cv} rounds.", session_outfile)

    for split in splits:
        cv_enabled = cv > 1
        for cv_round in range(0, cv):
            dataset_seed = RANDOM_SEED + cv_round if cv_enabled else RANDOM_SEED

            # The FND and LNK training sets are not allowed to have what is in the E2E test set
            e2e_train_ds, e2e_eval_ds, e2e_test_ds = split_dataset(e2e_vul4j_df, ratios=split, label_col=LABEL_COL, seed=dataset_seed)
            train_ratio_rescaled = float(split[0]) / (split[0] + split[1])
            # TODO Check corner cases, i.e., when split1 is 0, resulting in 1, which doesn't like. In other words, if it is 1, just don't do it

            text_set = set(e2e_test_ds[TEXT_1_COL])
            filtered_fnd_vul4j_df = DataFrame([row for row in fnd_vul4j_df.to_dict("records") if row[TEXT_COL] not in text_set])
            fnd_ds = datasets.Dataset.from_pandas(DataFrame(filtered_fnd_vul4j_df)).class_encode_column(LABEL_COL)
            if train_ratio_rescaled == 1.0:
                fnd_train_ds = copy.deepcopy(fnd_ds)
                fnd_eval_ds = None
            else:
                fnd_traineval_ds = fnd_ds.train_test_split(train_size=train_ratio_rescaled, stratify_by_column=LABEL_COL, seed=dataset_seed)
                fnd_train_ds = fnd_traineval_ds["train"]
                fnd_eval_ds = fnd_traineval_ds["test"]

            text_1_2_set = set(zip(e2e_test_ds[TEXT_1_COL], e2e_test_ds[TEXT_2_COL]))
            filtered_lnk_vul4j_df = DataFrame([row for row in lnk_vul4j_df.to_dict("records") if (row[TEXT_1_COL], row[TEXT_2_COL]) not in text_1_2_set])
            lnk_ds = datasets.Dataset.from_pandas(DataFrame(filtered_lnk_vul4j_df)).class_encode_column(LABEL_COL)
            if train_ratio_rescaled == 1.0:
                lnk_train_ds = copy.deepcopy(lnk_ds)
                lnk_eval_ds = None
            else:
                lnk_traineval_ds = lnk_ds.train_test_split(train_size=train_ratio_rescaled, stratify_by_column=LABEL_COL, seed=dataset_seed)
                lnk_train_ds = lnk_traineval_ds["train"]
                lnk_eval_ds = lnk_traineval_ds["test"]

            if any(e is None for e in [fnd_train_ds, lnk_train_ds, e2e_train_ds, e2e_test_ds]):
                print_stdout_file(f"Ratio {split} is improper for splitting the dataset as requested. Cannot proceed with this configuration.", session_outfile)
                continue
            if args.debug:
                print_stdout_file(f"THIS IS A DEBUG RUN: THE RESULTS ARE NOT MEANINGFUL!", session_outfile)
                fnd_train_ds = fnd_train_ds.train_test_split(train_size=0.008, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
                if fnd_eval_ds:
                    fnd_eval_ds = fnd_eval_ds.train_test_split(train_size=0.02, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
                lnk_train_ds = lnk_train_ds.train_test_split(train_size=0.01, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
                if lnk_eval_ds:
                    lnk_eval_ds = lnk_eval_ds.train_test_split(train_size=0.05, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
                e2e_train_ds = e2e_train_ds.train_test_split(train_size=0.008, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
                if e2e_eval_ds:
                    e2e_eval_ds = e2e_eval_ds.train_test_split(train_size=0.02, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
                e2e_test_ds = e2e_test_ds.train_test_split(train_size=0.05, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
            print_stdout_file(f"Training set size (Finder): {len(fnd_train_ds)}", session_outfile)
            print_stdout_file(f"- Nr. True instances: {sum(1 for ex in fnd_train_ds if ex[LABEL_COL] == 1)}", session_outfile)
            if fnd_eval_ds:
                print_stdout_file(f"Evaluation set size (Finder): {len(fnd_eval_ds)}", session_outfile)
                print_stdout_file(f"- Nr. True instances: {sum(1 for ex in fnd_eval_ds if ex[LABEL_COL] == 1)}", session_outfile)
            print_stdout_file(f"Training set size (Linker): {len(lnk_train_ds)}", session_outfile)
            print_stdout_file(f"- Nr. True instances: {sum(1 for ex in lnk_train_ds if ex[LABEL_COL] == 1)}", session_outfile)
            if lnk_eval_ds:
                print_stdout_file(f"Evaluation set size (Linker): {len(lnk_eval_ds)}", session_outfile)
                print_stdout_file(f"- Nr. True instances: {sum(1 for ex in lnk_eval_ds if ex[LABEL_COL] == 1)}", session_outfile)
            print_stdout_file(f"Training set size (E2E): {len(e2e_train_ds)}", session_outfile)
            print_stdout_file(f"- Nr. True instances: {sum(1 for ex in e2e_train_ds if ex[LABEL_COL] == 1)}", session_outfile)
            if e2e_eval_ds:
                print_stdout_file(f"Evaluation set size (E2E): {len(e2e_eval_ds)}", session_outfile)
                print_stdout_file(f"- Nr. True instances: {sum(1 for ex in e2e_eval_ds if ex[LABEL_COL] == 1)}", session_outfile)
            print_stdout_file(f"Test set size (E2E): {len(e2e_test_ds)}", session_outfile)
            print_stdout_file(f"- Nr. True instances: {sum(1 for ex in e2e_test_ds if ex[LABEL_COL] == 1)}", session_outfile)

            if model_id in NN_E2E_MODELS.keys() or model_id in LM_E2E_MODELS.keys():
                if model_id in NN_E2E_MODELS.keys():
                    model_class, export_dir = NN_E2E_MODELS[model_id]
                    experim_configs = get_nn_e2e_experiment_configs(args)
                    eval_fn = evaluate_nn_e2e
                elif model_id in LM_E2E_MODELS.keys():
                    model_class, export_dir = LM_E2E_MODELS[model_id]
                    experim_configs = get_lm_e2e_experiment_configs(args)
                    eval_fn = evaluate_lm_e2e

                if args.metric is None:
                    print_stdout_file("No metric to optimize specified. Using default 'loss'.", session_outfile)
                    metric = "loss"
                else:
                    metric = args.metric
                hf_cache_dir = args.cache
                if hf_cache_dir is None:
                    print_stdout_file("Cache directory for HuggingFace not specified. Using the default home directory.", session_outfile)
                else:
                    os.makedirs(hf_cache_dir, exist_ok=True)
                for model_config, hyperparam_space in experim_configs:
                    # NOTE We set this global variable to request the loading of Unsloth API. This is to prevent the case in which Unsloth is loaded but then not used in the rest of the code, resulting in a NotImplementedError. This is a ugly hotfix because Unsloth rewrites some scripts of Transformers API
                    global_vars.MUST_LOAD_UNSLOTH = UNSLOTH_TRAINING_KEY in model_config
                    full_config = {CommonConfigKeys.SPLIT: split, **model_config}
                    full_config_str = config_to_str(full_config)
                    hyperparam_str = config_to_str(hyperparam_space)
                    print_stdout_file(f"\nExperimenting with configuration '{full_config_str}'", session_outfile)
                    print_stdout_file(json.dumps(full_config, indent=2), session_outfile)
                    print_stdout_file(f"Searching in the following hyperparameter space '{hyperparam_str}'", session_outfile)
                    print_stdout_file(json.dumps(hyperparam_space, indent=2), session_outfile)
                    eval_run_info, _ = eval_fn(model_config=model_config,
                                               model_class=model_class,
                                               export_dir=export_dir,
                                               log_outfile=session_outfile,
                                               e2e_test_ds=e2e_test_ds,
                                               fnd_train_ds=fnd_train_ds,
                                               fnd_eval_ds=fnd_eval_ds,
                                               lnk_train_ds=lnk_train_ds,
                                               lnk_eval_ds=lnk_eval_ds,
                                               e2e_train_ds=e2e_train_ds,
                                               hyperparam_space=hyperparam_space,
                                               e2e_eval_ds=e2e_eval_ds,
                                               metric=metric,
                                               cache_dir=hf_cache_dir)
                    final_results = {
                        CommonConfigKeys.SPLIT: split,
                        CommonConfigKeys.DATASET_SEED: dataset_seed,
                        **eval_run_info
                    }
                    print_stdout_file(f"\nFinal results:", session_outfile)
                    print_stdout_file(final_results, session_outfile)
                    if cv_enabled:
                        full_config_str += f"_{cv_round}"
                    with open(os.path.join(session_outdir, f"{full_config_str}.json"), 'w') as fout:
                        json.dump(final_results, fp=fout, indent=2, cls=NpEncoder)
            elif model_id in [End2EndName.GREP_E2E, End2EndName.TERMS_E2E, End2EndName.SIM_E2E]:
                if model_id == End2EndName.GREP_E2E:
                    experim_configs_fn = get_grep_mtc_experiment_configs
                    eval_fn = eval_grep_mtc
                if model_id == End2EndName.TERMS_E2E:
                    experim_configs_fn = get_terms_mtc_experiment_configs
                    eval_fn = eval_terms_mtc
                if model_id == End2EndName.SIM_E2E:
                    experim_configs_fn = get_sim_mtc_experiment_configs
                    eval_fn = eval_sim_mtc
                experim_configs = experim_configs_fn(args)
                for model_config, hyperparam_space in experim_configs:
                    full_config = {CommonConfigKeys.SPLIT: split, **model_config}
                    full_config_str = config_to_str(full_config)
                    print_stdout_file(f"\nExperimenting with configuration '{full_config_str}'", session_outfile)
                    print_stdout_file(json.dumps(full_config, indent=2), session_outfile)
                    eval_run_info, _ = eval_fn(e2e_train_ds, e2e_test_ds, model_config, session_outfile, hyperparam_space)
                    final_results = {
                        CommonConfigKeys.SPLIT: split,
                        CommonConfigKeys.DATASET_SEED: dataset_seed,
                        **eval_run_info
                    }
                    print_stdout_file(f"\nFinal results:", session_outfile)
                    print_stdout_file(final_results, session_outfile)
                    if cv_enabled:
                        full_config_str += f"_{cv_round}"
                    with open(os.path.join(session_outdir, f"{full_config_str}.json"), 'w') as fout:
                        json.dump(final_results, fp=fout, indent=2, cls=NpEncoder)
            elif model_id == End2EndName.FIX_E2E:
                eval_run_info = eval_fix(e2e_test_ds, session_outfile)
                final_results = {
                    CommonConfigKeys.SPLIT: split,
                    CommonConfigKeys.DATASET_SEED: dataset_seed,
                    **eval_run_info
                }
                print_stdout_file(f"\nResults on test set:", session_outfile)
                print_stdout_file(final_results, session_outfile)
                out_filename = "test_results"
                if cv_enabled:
                    out_filename += f"_{cv_round}"
                with open(os.path.join(session_outdir, f"{out_filename}.json"), 'w') as fout:
                    json.dump(final_results, fp=fout, indent=2, cls=NpEncoder)
