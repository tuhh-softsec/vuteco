import datetime as dt
import json
import os
import sys
from argparse import ArgumentParser

import datasets
from evaluate import logging as el

from vuteco.core.common import global_vars
from vuteco.core.common.args_to_config import (
    get_grep_fnd_experiment_configs, get_grep_mtc_experiment_configs,
    get_lm_fnd_experiment_configs, get_lm_lnk_experiment_configs,
    get_nn_fnd_experiment_configs, get_nn_lnk_experiment_configs,
    get_sim_mtc_experiment_configs, get_terms_mtc_experiment_configs,
    get_vocab_fnd_experiment_configs)
from vuteco.core.common.cli_args import (add_base_args, add_config_args,
                                         parse_common_args)
from vuteco.core.common.cli_constants import (LM_FINDER_MODELS,
                                              LM_LINKER_MODELS,
                                              NN_FINDER_MODELS,
                                              NN_LINKER_MODELS)
from vuteco.core.common.constants import (EXPERIMENT_SESSION_DIRPATH,
                                          LABEL_COL, RANDOM_SEED,
                                          UNSLOTH_TRAINING_KEY,
                                          CommonConfigKeys, FinderName,
                                          LinkerName)
from vuteco.core.common.load_dataset import (load_vul4j_for_fnd,
                                             load_vul4j_for_lnk)
from vuteco.core.common.utils_training import (NpEncoder, print_stdout_file,
                                               split_dataset)
from vuteco.train.evaluation.evaluate_common import config_to_str
from vuteco.train.evaluation.evaluate_fix import eval_fix
from vuteco.train.evaluation.evaluate_grep_fnd import eval_grep_fnd
from vuteco.train.evaluation.evaluate_grep_mtc import eval_grep_mtc
from vuteco.train.evaluation.evaluate_lm import (evaluate_lm_fnd,
                                                 evaluate_lm_lnk)
from vuteco.train.evaluation.evaluate_nn import (evaluate_nn_fnd,
                                                 evaluate_nn_lnk)
from vuteco.train.evaluation.evaluate_sim_mtc import eval_sim_mtc
from vuteco.train.evaluation.evaluate_terms_mtc import eval_terms_mtc
from vuteco.train.evaluation.evaluate_vocab_fnd import eval_vocab_fnd

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not sys.stdout.isatty():
        datasets.disable_progress_bar()
        el.disable_progress_bar()

    argparser = ArgumentParser()
    argparser = add_base_args(argparser, "finder", "linker", "input", "output", "debug", "cache")
    argparser = add_config_args(argparser, "splits", "cv", "metric", "loss")
    argparser = add_config_args(argparser, "epochs", "hidden-size-1", "hidden-size-2", "augment", "merge", "one-line", "one-line-both", "use-cwe", "use-cwe-both")
    argparser = add_config_args(argparser, "unsloth-training-both", "unsloth-training")
    argparser = add_config_args(argparser, "separate-both", "separate-yes", "vocab-extractor", "vocab-matches", "grep-matches", "grep-extended-both", "grep-extended")
    argparser = add_config_args(argparser, "terms-threshold", "sim-extractor", "sim-threshold", "keywords-nr", "keywords-dedup")
    args = argparser.parse_args()

    # Finder model has the precedence
    if args.finder:
        model_id = args.finder
        input_load_fn = load_vul4j_for_fnd
    elif args.linker:
        model_id = args.linker
        input_load_fn = load_vul4j_for_lnk

    vul4j_df, session_outdir, session_outfile = parse_common_args(args, default_outdir=os.path.join(EXPERIMENT_SESSION_DIRPATH, model_id), input_load_fn=input_load_fn)
    print_stdout_file(f"[{dt.datetime.now()}] Starting training-validation-testing session of {model_id}", session_outfile)
    print_stdout_file("Distribution of the labels", session_outfile)
    print_stdout_file(vul4j_df[LABEL_COL].value_counts(), session_outfile)

    if args.metric is None:
        print_stdout_file("No metric to optimize specified. Using default 'loss'.", session_outfile)
        metric = "loss"
    else:
        metric = args.metric
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
            train_ds, eval_ds, test_ds = split_dataset(vul4j_df, ratios=split, label_col=LABEL_COL, seed=dataset_seed)
            if train_ds is None or test_ds is None:
                print_stdout_file(f"Ratio {split} is improper for splitting the dataset as requested. Cannot proceed with this configuration.", session_outfile)
                continue
            if args.debug:
                print_stdout_file(f"THIS IS A DEBUG RUN: THE RESULTS ARE NOT MEANINGFUL!", session_outfile)
                if "fnd" in model_id:
                    debug_train_size = 0.008
                else:
                    debug_train_size = 0.01
                debug_eval_size = 0.05
                debug_test_size = 0.05
                train_ds = train_ds.train_test_split(train_size=debug_train_size, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
                if eval_ds:
                    eval_ds = eval_ds.train_test_split(train_size=debug_eval_size, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
                test_ds = test_ds.train_test_split(train_size=debug_test_size, stratify_by_column=LABEL_COL, seed=dataset_seed)["train"]
            print_stdout_file(f"Training set size: {len(train_ds)}", session_outfile)
            print_stdout_file(f"- Nr. True instances: {sum(1 for ex in train_ds if ex[LABEL_COL] == 1)}", session_outfile)
            if eval_ds:
                print_stdout_file(f"Evaluation set size: {len(eval_ds)}", session_outfile)
                print_stdout_file(f"- Nr. True instances: {sum(1 for ex in eval_ds if ex[LABEL_COL] == 1)}", session_outfile)
            print_stdout_file(f"Test set size: {len(test_ds)}", session_outfile)
            print_stdout_file(f"- Nr. True instances: {sum(1 for ex in test_ds if ex[LABEL_COL] == 1)}", session_outfile)

            hf_cache_dir = args.cache
            common_eval_args = {
                "train_ds": train_ds,
                "test_ds": test_ds,
                "log_outfile": session_outfile
            }

            if model_id == FinderName.FIX_FND:
                eval_run_info = eval_fix(test_ds, session_outfile)
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
                continue
            
            if model_id in NN_FINDER_MODELS.keys() or model_id in LM_FINDER_MODELS.keys() or model_id in NN_LINKER_MODELS.keys() or model_id in LM_LINKER_MODELS.keys():
                if model_id in NN_FINDER_MODELS.keys():
                    model_class, export_dir = NN_FINDER_MODELS[model_id]
                    experim_configs = get_nn_fnd_experiment_configs(args)
                    eval_fn = evaluate_nn_fnd
                elif model_id in LM_FINDER_MODELS.keys():
                    model_class, export_dir = LM_FINDER_MODELS[model_id]
                    experim_configs = get_lm_fnd_experiment_configs(args)
                    eval_fn = evaluate_lm_fnd
                elif model_id in NN_LINKER_MODELS.keys():
                    model_class, export_dir = NN_LINKER_MODELS[model_id]
                    experim_configs = get_nn_lnk_experiment_configs(args)
                    eval_fn = evaluate_nn_lnk
                elif model_id in LM_LINKER_MODELS.keys():
                    model_class, export_dir = LM_LINKER_MODELS[model_id]
                    experim_configs = get_lm_lnk_experiment_configs(args)
                    eval_fn = evaluate_lm_lnk

                if hf_cache_dir is None:
                    print_stdout_file("Cache directory for HuggingFace not specified. Using the default home directory.", session_outfile)
                else:
                    os.makedirs(hf_cache_dir, exist_ok=True)
                common_eval_args = {
                    "model_class": model_class,
                    "export_dir": export_dir,
                    "cache_dir": hf_cache_dir,
                    "eval_ds": eval_ds,
                    "metric": metric,
                    **common_eval_args
                }
            elif model_id == FinderName.VOCABULARY_FND:
                experim_configs = get_vocab_fnd_experiment_configs(args)
                eval_fn = eval_vocab_fnd
            elif model_id == FinderName.GREP_FND:
                experim_configs = get_grep_fnd_experiment_configs(args)
                eval_fn = eval_grep_fnd
                common_eval_args = {
                    "eval_ds": eval_ds,
                    **common_eval_args
                }
            elif model_id == LinkerName.GREP_LNK:
                experim_configs = get_grep_mtc_experiment_configs(args)
                eval_fn = eval_grep_mtc
                common_eval_args = {
                    "eval_ds": eval_ds,
                    **common_eval_args
                }
            elif model_id == LinkerName.TERMS_LNK:
                experim_configs = get_terms_mtc_experiment_configs(args)
                eval_fn = eval_terms_mtc
            elif model_id == LinkerName.SIM_LNK:
                experim_configs = get_sim_mtc_experiment_configs(args)
                eval_fn = eval_sim_mtc
                common_eval_args = {
                    "cache_dir": hf_cache_dir,
                    **common_eval_args
                }

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
                eval_args = {
                    **common_eval_args,
                    "hyperparam_space": hyperparam_space
                }
                eval_run_info, _ = eval_fn(**{k: v for k, v in eval_args.items() if v is not None},
                                           model_config=model_config)
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
