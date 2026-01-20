import datetime as dt
import os
import sys
from argparse import ArgumentParser

import datasets
from cli_constants import (LM_FINDER_MODELS, LM_LINKER_MODELS,
                           NN_FINDER_MODELS, NN_LINKER_MODELS)
from common import global_vars
from common.args_to_config import (get_lm_fnd_config, get_lm_lnk_config,
                                   get_nn_fnd_config, get_nn_lnk_config)
from common.cli_args import add_base_args, add_config_args, parse_common_args
from common.constants import (LABEL_COL, RANDOM_SEED, TRAINED_MODEL_DIRPATH,
                              UNSLOTH_TRAINING_KEY, FinderName, LinkerName)
from common.load_dataset import load_vul4j_for_fnd, load_vul4j_for_lnk
from common.utils_training import split_dataset
from evaluate import logging as el
from training.fit_vocab_fnd import fit_vocab_fnd, get_vocab_fnd_config
from training.save_grep_fnd import get_grep_fnd_config, save_grep_fnd
from training.save_grep_mtc import get_grep_mtc_config, save_grep_mtc
from training.save_sim_mtc import get_sim_mtc_config, save_sim_mtc
from training.save_terms_mtc import get_terms_mtc_config, save_terms_mtc
from training.train_lm import train_lm_fnd, train_lm_lnk
from training.train_nn import train_nn_fnd, train_nn_lnk

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not sys.stdout.isatty():
        datasets.disable_progress_bar()
        el.disable_progress_bar()

    argparser = ArgumentParser()
    argparser = add_base_args(argparser, "finder", "linker", "input", "output", "debug", "cache")
    argparser = add_config_args(argparser, "splits", "metric", "loss")
    argparser = add_config_args(argparser, "epochs", "hidden-size-1", "hidden-size-2", "augment", "merge", "one-line", "use-cwe")
    argparser = add_config_args(argparser, "unsloth-training")
    argparser = add_config_args(argparser, "separate", "vocab-matches", "grep-matches", "grep-extended-both", "grep-extended", "terms-threshold", "sim-extractor", "sim-threshold", "keywords-nr", "keywords-dedup")
    args = argparser.parse_args()

    # Finder model has the precedence
    if args.finder:
        model_id = args.finder
        input_load_fn = load_vul4j_for_fnd
    elif args.linker:
        model_id = args.linker
        input_load_fn = load_vul4j_for_lnk

    vul4j_df, out_dirpath = parse_common_args(args, default_outdir=os.path.join(TRAINED_MODEL_DIRPATH, model_id), input_load_fn=input_load_fn, start_session=False)
    print("Distribution of the labels")
    print(vul4j_df[LABEL_COL].value_counts())

    if args.metric is None:
        print("No metric to optimize specified. Using default 'loss'.")
        metric = "loss"
    else:
        metric = args.metric
    if args.splits is None:
        print("No dataset splits specified. Using default [0.85, 0.15, 0.0].")
        split = [0.85, 0.15, 0.0]
    else:
        split = [float(v) for v in args.splits.split("-")]
    print(f"[{dt.datetime.now()}] Starting exporting {model_id}")
    train_ds, eval_ds, _ = split_dataset(vul4j_df, ratios=split, label_col=LABEL_COL, seed=RANDOM_SEED)
    if args.debug:
        print(f"THIS IS A DEBUG RUN: ANY MODEL TRAINED WILL NOT BE MEANINGFUL!")
        train_ds = train_ds.train_test_split(train_size=150, stratify_by_column=LABEL_COL, seed=RANDOM_SEED)["train"]
        if eval_ds:
            eval_ds = eval_ds.train_test_split(train_size=50, stratify_by_column=LABEL_COL, seed=RANDOM_SEED)["train"]
    print(f"Training set size: {len(train_ds)}")
    print(f"- Nr. True instances: {sum(1 for ex in train_ds if ex[LABEL_COL] == 1)}")
    if eval_ds:
        print(f"Evaluation set size: {len(eval_ds)}")
        print(f"- Nr. True instances: {sum(1 for ex in eval_ds if ex[LABEL_COL] == 1)}")

    hf_cache_dir = args.cache
    if model_id in NN_FINDER_MODELS.keys() or model_id in LM_FINDER_MODELS.keys() or model_id in NN_LINKER_MODELS.keys() or model_id in LM_LINKER_MODELS.keys():
        if model_id in NN_FINDER_MODELS.keys():
            model_class, _ = NN_FINDER_MODELS[model_id]
            config, hp_config = get_nn_fnd_config(args)
            train_fn = train_nn_fnd
        if model_id in LM_FINDER_MODELS.keys():
            model_class, _ = LM_FINDER_MODELS[model_id]
            config, hp_config = get_lm_fnd_config(args)
            train_fn = train_lm_fnd
        if model_id in NN_LINKER_MODELS.keys():
            model_class, _ = NN_LINKER_MODELS[model_id]
            config, hp_config = get_nn_lnk_config(args)
            train_fn = train_nn_lnk
        if model_id in LM_LINKER_MODELS.keys():
            model_class, _ = LM_LINKER_MODELS[model_id]
            config, hp_config = get_lm_lnk_config(args)
            train_fn = train_lm_lnk
        
        if hf_cache_dir is None:
            print("Cache directory for HuggingFace not specified. Using the default home directory.")
        else:
            os.makedirs(hf_cache_dir, exist_ok=True)
        # NOTE We set this global variable to request the loading of Unsloth API. This is to prevent the case in which Unsloth is loaded but then not used in the rest of the code, resulting in a NotImplementedError. This is a ugly hotfix because Unsloth rewrites some scripts of Transformers API
        global_vars.MUST_LOAD_UNSLOTH = UNSLOTH_TRAINING_KEY in config
        train_fn(train_ds, model_class, model_config=config, export_dir=out_dirpath, cache_dir=hf_cache_dir, hyperparam_config=hp_config, eval_ds=eval_ds, metric=metric)
    elif model_id == FinderName.VOCABULARY_FND:
        config, hp_config = get_vocab_fnd_config(args)
        fit_vocab_fnd(train_ds, config, export_dir=out_dirpath, hyperparam_config=hp_config)
    elif model_id == FinderName.GREP_FND:
        config, hp_config = get_grep_fnd_config(args)
        save_grep_fnd(config, export_dir=out_dirpath, hyperparam_config=hp_config)
    elif model_id == LinkerName.GREP_LNK:
        config, hp_config = get_grep_mtc_config(args)
        save_grep_mtc(config, export_dir=out_dirpath, hyperparam_config=hp_config)
    elif model_id == LinkerName.TERMS_LNK:
        config, hp_config = get_terms_mtc_config(args)
        save_terms_mtc(config, export_dir=out_dirpath, hyperparam_config=hp_config)
    elif model_id == LinkerName.SIM_LNK:
        config, hp_config = get_sim_mtc_config(args)
        save_sim_mtc(config, export_dir=out_dirpath, hyperparam_config=hp_config, cache_dir=hf_cache_dir)
