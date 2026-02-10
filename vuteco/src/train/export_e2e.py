import datetime as dt
import os
import sys
from argparse import ArgumentParser

import datasets
import numpy as np
from core.common import global_vars
from core.common.args_to_config import get_lm_e2e_config, get_nn_e2e_config
from core.common.cli_args import add_base_args, add_config_args
from core.common.cli_constants import LM_E2E_MODELS, NN_E2E_MODELS
from core.common.constants import (DATA_DIRPATH, DEVICE, LABEL_COL,
                                   RANDOM_SEED, TRAINED_MODEL_DIRPATH,
                                   UNSLOTH_TRAINING_KEY, VUL4J_TEST_FILEPATH,
                                   E2ETrainingType, MergeStyle)
from core.common.load_dataset import (load_vul4j_for_e2e, load_vul4j_for_fnd,
                                      load_vul4j_for_lnk)
from core.common.utils_training import split_dataset
from core.modeling.modeling_lm_e2e import (LanguageModelE2EConfig,
                                           LanguageModelE2EConfigKeys)
from core.modeling.modeling_nn_e2e import (NeuralNetworkE2EConfig,
                                           NeuralNetworkE2EConfigKeys)
from core.modeling.modeling_nn_fnd import NeuralNetworkFinderConfigKeys
from core.modeling.modeling_nn_lnk import NeuralNetworkLinkerConfigKeys
from evaluate import logging as el
from train.evaluation import evaluate_lm, evaluate_nn
from sklearn.utils import compute_class_weight
from train.training.train_lm import train_lm_e2e, train_lm_fnd, train_lm_lnk
from train.training.train_nn import train_nn_e2e, train_nn_fnd, train_nn_lnk

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
    argparser = add_config_args(argparser, "unsloth-training")
    argparser = add_config_args(argparser, "fnd-epochs", "fnd-hidden-size-1", "fnd-hidden-size-2", "fnd-augment", "fnd-loss", "fnd-one-line")
    argparser = add_config_args(argparser, "lnk-epochs", "lnk-hidden-size-1", "lnk-hidden-size-2", "lnk-augment", "lnk-loss", "lnk-one-line")
    argparser = add_config_args(argparser, "use-cwe", "merge")
    args = argparser.parse_args()

    model_id = args.end_to_end

    default_outdir = os.path.join(TRAINED_MODEL_DIRPATH, model_id)
    if args.output is None:
        print(f"Directory where to write the output results not supplied. Using the default location {default_outdir}.")
        out_dirpath = os.path.abspath(default_outdir)
    else:
        out_dirpath = os.path.abspath(args.output)
    if args.input is None:
        print(f"Directory containing the Vul4J dataset not supplied. Using the default expected location in \"{DATA_DIRPATH}\"")
        dataset_basepath = os.path.abspath(DATA_DIRPATH)
    else:
        dataset_basepath = os.path.abspath(args.input)

    print("Loading Vul4J Dataset for End-to-End model...")
    e2e_vul4j_df = load_vul4j_for_e2e(VUL4J_TEST_FILEPATH, dataset_basepath)
    fnd_vul4j_df = load_vul4j_for_fnd(VUL4J_TEST_FILEPATH, dataset_basepath)
    lnk_vul4j_df = load_vul4j_for_lnk(VUL4J_TEST_FILEPATH, dataset_basepath)
    if any(e is None for e in [fnd_vul4j_df, lnk_vul4j_df]):
        print("Failed to load Vul4J dataset. Exiting")
        exit(1)
    print("Distribution of the labels in E2E dataset")
    print(e2e_vul4j_df[LABEL_COL].value_counts())

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
    e2e_train_ds, e2e_eval_ds, _ = split_dataset(e2e_vul4j_df, ratios=split, label_col=LABEL_COL, seed=RANDOM_SEED)
    # if e2e_test_ds:
    #    text_set = set(e2e_test_ds[TEXT_1_COL])
    #    fnd_vul4j_df = DataFrame([row for row in fnd_vul4j_df.to_dict("records") if row[TEXT_COL] not in text_set])
    #    text_1_2_set = set(zip(e2e_test_ds[TEXT_1_COL], e2e_test_ds[TEXT_2_COL]))
    #    lnk_vul4j_df = DataFrame([row for row in lnk_vul4j_df.to_dict("records") if (row[TEXT_1_COL], row[TEXT_2_COL]) not in text_1_2_set])
    # fnd_train_ds = datasets.Dataset.from_pandas(DataFrame(fnd_vul4j_df)).class_encode_column(LABEL_COL)
    # lnk_train_ds = datasets.Dataset.from_pandas(DataFrame(lnk_vul4j_df)).class_encode_column(LABEL_COL)
    fnd_train_ds, fnd_eval_ds, _ = split_dataset(fnd_vul4j_df, ratios=split, label_col=LABEL_COL, seed=RANDOM_SEED)
    lnk_train_ds, lnk_eval_ds, _ = split_dataset(lnk_vul4j_df, ratios=split, label_col=LABEL_COL, seed=RANDOM_SEED)

    if args.debug:
        print(f"THIS IS A DEBUG RUN: ANY MODEL TRAINED WILL NOT BE MEANINGFUL!")
        fnd_train_ds = fnd_train_ds.train_test_split(train_size=300, stratify_by_column=LABEL_COL, seed=RANDOM_SEED)["train"]
        if fnd_eval_ds:
            fnd_eval_ds = fnd_eval_ds.train_test_split(train_size=100, stratify_by_column=LABEL_COL, seed=RANDOM_SEED)["train"]
        lnk_train_ds = lnk_train_ds.train_test_split(train_size=300, stratify_by_column=LABEL_COL, seed=RANDOM_SEED)["train"]
        if lnk_eval_ds:
            lnk_eval_ds = lnk_eval_ds.train_test_split(train_size=100, stratify_by_column=LABEL_COL, seed=RANDOM_SEED)["train"]
        e2e_train_ds = e2e_train_ds.train_test_split(train_size=500, stratify_by_column=LABEL_COL, seed=RANDOM_SEED)["train"]
        if e2e_eval_ds:
            e2e_eval_ds = e2e_eval_ds.train_test_split(train_size=100, stratify_by_column=LABEL_COL, seed=RANDOM_SEED)["train"]
    print(f"Training set size (Finder): {len(fnd_train_ds)}")
    print(f"- Nr. True instances: {sum(1 for ex in fnd_train_ds if ex[LABEL_COL] == 1)}")
    if fnd_eval_ds:
        print(f"Evaluation set size: {len(fnd_eval_ds)}")
        print(f"- Nr. True instances: {sum(1 for ex in fnd_eval_ds if ex[LABEL_COL] == 1)}")
    print(f"Training set size (Linker): {len(lnk_train_ds)}")
    print(f"- Nr. True instances: {sum(1 for ex in lnk_train_ds if ex[LABEL_COL] == 1)}")
    if lnk_eval_ds:
        print(f"Evaluation set size: {len(lnk_eval_ds)}")
        print(f"- Nr. True instances: {sum(1 for ex in lnk_eval_ds if ex[LABEL_COL] == 1)}")
    print(f"Training set size (E2E): {len(e2e_train_ds)}")
    print(f"- Nr. True instances: {sum(1 for ex in e2e_train_ds if ex[LABEL_COL] == 1)}")
    if e2e_eval_ds:
        print(f"Evaluation set size: {len(e2e_eval_ds)}")
        print(f"- Nr. True instances: {sum(1 for ex in e2e_eval_ds if ex[LABEL_COL] == 1)}")

    if model_id in NN_E2E_MODELS.keys() or model_id in LM_E2E_MODELS.keys():
        if model_id in NN_E2E_MODELS.keys():
            model_class, _ = NN_E2E_MODELS[model_id]
            e2e_model_config, e2e_hp_config = get_nn_e2e_config(args)
            train_fnd_fn = train_nn_fnd
            train_lnk_fn = train_nn_lnk
            train_e2e_fn = train_nn_e2e
            key_class = NeuralNetworkE2EConfigKeys
            config_class = NeuralNetworkE2EConfig
            fnd_config, fnd_hp = evaluate_nn.e2e_to_fnd_config(e2e_model_config, e2e_hp_config)
            lnk_config, lnk_hp = evaluate_nn.e2e_to_lnk_config(e2e_model_config, e2e_hp_config)
            e2e_config_args = {
                "fnd_clf_hidden_size_1": e2e_hp_config[NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_1],
                "fnd_clf_hidden_size_2": e2e_hp_config[NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_2],
                "fnd_loss_fun": e2e_model_config[NeuralNetworkE2EConfigKeys.FND_LOSS],
                "fnd_class_weights": compute_class_weight(class_weight="balanced", classes=np.unique(fnd_train_ds[LABEL_COL]), y=fnd_train_ds[LABEL_COL]).tolist(),
                "lnk_clf_hidden_size_1": e2e_hp_config[NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_1],
                "lnk_clf_hidden_size_2": e2e_hp_config[NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_2],
                "lnk_loss_fun": e2e_model_config[NeuralNetworkE2EConfigKeys.LNK_LOSS],
                "lnk_class_weights": compute_class_weight(class_weight="balanced", classes=np.unique(lnk_train_ds[LABEL_COL]), y=lnk_train_ds[LABEL_COL]).tolist(),
                "fnd_one_line_text": e2e_model_config[NeuralNetworkE2EConfigKeys.FND_ONE_LINE],
                "lnk_one_line_text": e2e_model_config[NeuralNetworkE2EConfigKeys.LNK_ONE_LINE],
                "use_cwe": e2e_model_config[NeuralNetworkLinkerConfigKeys.USE_CWE],
                "concat_test_first": e2e_model_config[NeuralNetworkE2EConfigKeys.MERGE] == MergeStyle.CONCAT_TEST_DESCR,
                "as_javadoc": e2e_model_config[NeuralNetworkE2EConfigKeys.MERGE] == MergeStyle.JAVADOC,
                "merge_embeddings": e2e_model_config[NeuralNetworkE2EConfigKeys.MERGE] == MergeStyle.LEARN,
                "archi_style": e2e_model_config[NeuralNetworkE2EConfigKeys.ARCHI_STYLE],
                "suspect_threshold": e2e_model_config[NeuralNetworkE2EConfigKeys.SUSPECT_THRESHOLD]
            }
        elif model_id in LM_E2E_MODELS.keys():
            model_class, _ = LM_E2E_MODELS[model_id]
            e2e_model_config, e2e_hp_config = get_lm_e2e_config(args)
            train_fnd_fn = train_lm_fnd
            train_lnk_fn = train_lm_lnk
            train_e2e_fn = train_lm_e2e
            key_class = LanguageModelE2EConfigKeys
            config_class = LanguageModelE2EConfig
            # NOTE We set this global variable to request the loading of Unsloth API. This is to prevent the case in which Unsloth is loaded but then not used in the rest of the code, resulting in a NotImplementedError. This is a ugly hotfix because Unsloth rewrites some scripts of Transformers API
            global_vars.MUST_LOAD_UNSLOTH = UNSLOTH_TRAINING_KEY in e2e_model_config
            fnd_config, fnd_hp = None, None
            lnk_config, lnk_hp = evaluate_lm.e2e_to_lnk_config(e2e_model_config, e2e_hp_config)
            e2e_config_args = {
                "unsloth_training": e2e_model_config[LanguageModelE2EConfigKeys.UNSLOTH_TRAINING],
                "use_cwe": e2e_model_config[LanguageModelE2EConfigKeys.USE_CWE],
                "archi_style": e2e_model_config[LanguageModelE2EConfigKeys.ARCHI_STYLE],
            }
        e2e_config = config_class(**{k: v for k, v in e2e_config_args.items() if v is not None})
        hf_cache_dir = args.cache
        if hf_cache_dir is None:
            print("Cache directory for HuggingFace not specified. Using the default home directory.")
        else:
            os.makedirs(hf_cache_dir, exist_ok=True)
        if e2e_model_config[key_class.TRAIN_TYPE] in [E2ETrainingType.PRETRAIN_ONLY, E2ETrainingType.PRETRAIN_FINETUNE]:
            if fnd_config is not None and fnd_hp is not None:
                print(f"[{dt.datetime.now()}] Going to train the Finder module")
                fnd_export_dir = os.path.join(out_dirpath, "fnd")
                fnd_model, *_ = train_fnd_fn(train_ds=fnd_train_ds,
                                             model_class=model_class.FND_MODEL_CLASS,
                                             model_config=fnd_config,
                                             export_dir=fnd_export_dir,
                                             export_at_end=False,
                                             cache_dir=hf_cache_dir,
                                             hyperparam_config=fnd_hp,
                                             eval_ds=fnd_eval_ds,
                                             metric=metric
                                             )
            else:
                fnd_model = None
            print(f"[{dt.datetime.now()}] Going to train the Linker module")
            lnk_export_dir = os.path.join(out_dirpath, "lnk")
            lnk_model, *_ = train_lnk_fn(train_ds=lnk_train_ds,
                                         model_class=model_class.LNK_MODEL_CLASS,
                                         model_config=lnk_config,
                                         export_dir=lnk_export_dir,
                                         export_at_end=False,
                                         cache_dir=hf_cache_dir,
                                         hyperparam_config=lnk_hp,
                                         eval_ds=lnk_eval_ds,
                                         metric=metric
                                         )
            args = {"fnd_model": fnd_model} if fnd_model is not None else {}
            e2e_model = model_class(e2e_config, **args, lnk_model=lnk_model, cache_dir=hf_cache_dir)
        else:
            print(f"[{dt.datetime.now()}] Loading Finder and/or Linker with default weights")
            e2e_model = model_class(e2e_config, cache_dir=hf_cache_dir)

        e2e_model.to(DEVICE)
        e2e_export_dir = os.path.join(out_dirpath, "e2e")
        if e2e_model_config[key_class.TRAIN_TYPE] in [E2ETrainingType.FINETUNE_ONLY, E2ETrainingType.PRETRAIN_FINETUNE]:
            print(f"[{dt.datetime.now()}] Going to train the End-to-end module")
            train_e2e_fn(train_ds=e2e_train_ds,
                         model_class=model_class,
                         start_model=e2e_model,
                         model_config=e2e_model_config,
                         export_dir=e2e_export_dir,
                         cache_dir=hf_cache_dir,
                         hyperparam_config=e2e_hp_config,
                         eval_ds=e2e_eval_ds,
                         metric=metric
                         )
        else:
            e2e_model.save_pretrained(e2e_export_dir)
