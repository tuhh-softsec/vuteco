import datetime as dt
import os
import shutil
import sys
from typing import Any, Type

import numpy as np
import torch
from common.augment_data import augment_ds
from common.constants import (DETERMINISM, DEVICE, FINAL, LABEL_COL,
                              RANDOM_SEED, MergeStyle)
from common.utils_training import (LoggingCallback, clear_up,
                                   compute_eval_metrics, print_stdout_file)
from datasets import Dataset
from modeling.modeling_nn_e2e import (NeuralNetworkE2E, NeuralNetworkE2EConfig,
                                      NeuralNetworkE2EConfigKeys)
from modeling.modeling_nn_fnd import (NeuralNetworkFinder,
                                      NeuralNetworkFinderConfig,
                                      NeuralNetworkFinderConfigKeys)
from modeling.modeling_nn_lnk import (NeuralNetworkLinker,
                                      NeuralNetworkLinkerConfig,
                                      NeuralNetworkLinkerConfigKeys)
from sklearn.utils import compute_class_weight
from transformers import (IntervalStrategy, PreTrainedModel, Trainer,
                          TrainingArguments)


def run_nn_training(model: PreTrainedModel,
                    epochs: int,
                    tok_train_ds: Dataset,
                    tok_eval_ds: Dataset = None,
                    log_outfile: str = None,
                    metric: str = "loss",
                    export_dir: str = None,
                    export_at_end: bool = True) -> tuple[PreTrainedModel, Trainer, dt.timedelta, dict[str, float]]:
    os.makedirs(export_dir, exist_ok=True)
    training_args = TrainingArguments(
        # Training hyperparams
        lr_scheduler_type="linear",
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="adamw_torch",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        # When to run the evaluation on the evaluation set
        eval_strategy=IntervalStrategy.EPOCH if tok_eval_ds else IntervalStrategy.NO,
        # When to log
        logging_strategy=IntervalStrategy.EPOCH,
        # Log every 10% of the training. By default, if the eval_strategy is STEPS, then eval_steps will be set to logging_steps, so evaluation will also be done every 10% of the training
        # logging_steps=0.1,
        # Extra logging step at the start of the training
        logging_first_step=True,
        # When load_best_model_at_end is True, save_strategy must be like eval_strategy
        save_strategy=IntervalStrategy.EPOCH if tok_eval_ds and export_dir else IntervalStrategy.NO,
        # Keep just the best checkpoint
        save_total_limit=1,
        save_safetensors=False,
        # The metric for which the "best" is determined
        metric_for_best_model=metric,
        # The selected metric will be maximized by default ("True"), unless the metric ends with 'loss' substring"
        # greater_is_better=True,
        # Save a checkpoint every 10% of the training
        # save_steps=0.1,
        # Where to save the checkpoints
        output_dir=export_dir,
        # Load the best checkpoint at the end of the training
        load_best_model_at_end=tok_eval_ds is not None and export_dir is not None,
        # Other config
        disable_tqdm=not sys.stdout.isatty(),
        full_determinism=DETERMINISM,
        seed=RANDOM_SEED if DETERMINISM else False,
        data_seed=RANDOM_SEED if DETERMINISM else False,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train_ds,
        eval_dataset=tok_eval_ds,
        processing_class=model.tokenizer,
        compute_metrics=compute_eval_metrics,
        callbacks=[LoggingCallback(log_outfile)]
    )
    train_start = dt.datetime.now()
    print_stdout_file(f"[{train_start}] Training started!", log_outfile)
    trainer.train()
    clear_up()
    train_end = dt.datetime.now()
    train_duration = train_end - train_start
    print_stdout_file(f"[{train_end}] Training finished! (Total duration: {train_duration})", log_outfile)
    
    print_stdout_file(f"[{dt.datetime.now()}] Final evaluation started!", log_outfile)
    dev_results = trainer.evaluate(tok_eval_ds) if tok_eval_ds else None
    print_stdout_file(f"[{dt.datetime.now()}] Final evaluation finished!", log_outfile)
    if trainer.state.best_model_checkpoint and os.path.exists(trainer.state.best_model_checkpoint):
        shutil.rmtree(trainer.state.best_model_checkpoint)
    if export_dir and export_at_end:
        trainer.save_model(os.path.join(export_dir))
    return model, trainer, train_duration, dev_results


def train_nn_fnd(train_ds: Dataset,
                 model_class: Type[NeuralNetworkFinder],
                 start_model: NeuralNetworkFinder = None,
                 model_config: dict[NeuralNetworkFinderConfigKeys, Any] = None,
                 export_dir: str = None,
                 export_at_end: bool = True,
                 cache_dir: str = None,
                 hyperparam_config: dict[NeuralNetworkFinderConfigKeys, Any] = None,
                 eval_ds: Dataset = None,
                 metric: str = "loss",
                 log_outfile: str = None) -> tuple[NeuralNetworkFinder, Trainer, dt.timedelta, dict[str, float]]:
    old_train_size = len(train_ds)
    actual_train_ds = augment_ds(train_ds, model_config[NeuralNetworkFinderConfigKeys.AUGMENT_TECH], hyperparam_config[NeuralNetworkFinderConfigKeys.AUGMENT_EXT], log_outfile)
    if len(actual_train_ds) > old_train_size:
        print_stdout_file(f"New training set size: {old_train_size} -> {len(actual_train_ds)}", log_outfile)
    if start_model:
        nn_fnd_model = start_model
    else:
        fnd_config_args = {
            "clf_hidden_size_1": hyperparam_config[NeuralNetworkFinderConfigKeys.HIDDEN_SIZE_1],
            "clf_hidden_size_2": hyperparam_config[NeuralNetworkFinderConfigKeys.HIDDEN_SIZE_2],
            "loss_fun": model_config[NeuralNetworkFinderConfigKeys.LOSS],
            "class_weights": compute_class_weight(class_weight="balanced", classes=np.unique(actual_train_ds[LABEL_COL]), y=actual_train_ds[LABEL_COL]).tolist(),
            "one_line_text": model_config[NeuralNetworkFinderConfigKeys.ONE_LINE]
        }
        nn_fnd_model = model_class(
            NeuralNetworkFinderConfig(**{k: v for k, v in fnd_config_args.items() if v is not None}),
            cache_dir=cache_dir
        )
    nn_fnd_model.to(DEVICE)
    print_stdout_file(f"Training on device: {DEVICE}", log_outfile)

    # Tokenize instances
    print_stdout_file(f"[{dt.datetime.now()}] Preparing training instances", log_outfile)
    tok_actual_train_ds = actual_train_ds.map(nn_fnd_model.encode_batch, batched=True, batch_size=10**3, desc=f"Preparing {len(actual_train_ds)} training instances")
    if eval_ds:
        print_stdout_file(f"[{dt.datetime.now()}] Preparing evaluation instances", log_outfile)
        tok_eval_ds = eval_ds.map(nn_fnd_model.encode_batch, batched=True, batch_size=10**2, desc=f"Preparing {len(eval_ds)} validation instances")
    else:
        tok_eval_ds = None
    print_stdout_file(f"[{dt.datetime.now()}] Preparation done!", log_outfile)
    return run_nn_training(model=nn_fnd_model,
                           epochs=hyperparam_config[NeuralNetworkFinderConfigKeys.EPOCHS],
                           tok_train_ds=tok_actual_train_ds,
                           tok_eval_ds=tok_eval_ds,
                           log_outfile=log_outfile,
                           export_dir=export_dir,
                           metric=metric,
                           export_at_end=export_at_end)


def train_nn_lnk(train_ds: Dataset,
                 model_class: Type[NeuralNetworkLinker],
                 start_model: NeuralNetworkLinker = None,
                 model_config: dict[NeuralNetworkLinkerConfigKeys, Any] = None,
                 export_dir: str = None,
                 export_at_end: bool = True,
                 cache_dir: str = None,
                 hyperparam_config: dict[NeuralNetworkLinkerConfigKeys, Any] = None,
                 eval_ds: Dataset = None,
                 metric: str = "loss",
                 log_outfile: str = None) -> tuple[NeuralNetworkLinker, Trainer, dt.timedelta, dict, dict[str, float]]:
    old_train_size = len(train_ds)
    actual_train_ds = augment_ds(train_ds, model_config[NeuralNetworkLinkerConfigKeys.AUGMENT_TECH], hyperparam_config[NeuralNetworkLinkerConfigKeys.AUGMENT_EXT], log_outfile)
    if len(actual_train_ds) > old_train_size:
        print_stdout_file(f"New training set size: {old_train_size} -> {len(actual_train_ds)}", log_outfile)
    if start_model:
        nn_lnk_model = start_model
    else:
        lnk_config_args = {
            "clf_hidden_size_1": hyperparam_config[NeuralNetworkLinkerConfigKeys.HIDDEN_SIZE_1],
            "clf_hidden_size_2": hyperparam_config[NeuralNetworkLinkerConfigKeys.HIDDEN_SIZE_2],
            "loss_fun": model_config[NeuralNetworkLinkerConfigKeys.LOSS],
            "class_weights": compute_class_weight(class_weight="balanced", classes=np.unique(actual_train_ds[LABEL_COL]), y=actual_train_ds[LABEL_COL]).tolist(),
            "one_line_text": model_config[NeuralNetworkLinkerConfigKeys.ONE_LINE],
            "use_cwe": model_config[NeuralNetworkLinkerConfigKeys.USE_CWE],
            "concat_test_first": model_config[NeuralNetworkLinkerConfigKeys.MERGE] == MergeStyle.CONCAT_TEST_DESCR,
            "as_javadoc": model_config[NeuralNetworkLinkerConfigKeys.MERGE] == MergeStyle.JAVADOC,
            "merge_embeddings": model_config[NeuralNetworkLinkerConfigKeys.MERGE] == MergeStyle.LEARN
        }
        nn_lnk_model = model_class(
            NeuralNetworkLinkerConfig(**{k: v for k, v in lnk_config_args.items() if v is not None}),
            cache_dir=cache_dir
        )
    nn_lnk_model.to(DEVICE)
    print_stdout_file(f"Training on device: {DEVICE}", log_outfile)

    print_stdout_file(f"[{dt.datetime.now()}] Preparing training instances", log_outfile)
    tok_actual_train_ds = actual_train_ds.map(nn_lnk_model.encode_batch, batched=True, batch_size=10**3, desc=f"Preparing {len(actual_train_ds)} training instances")
    if eval_ds:
        print_stdout_file(f"[{dt.datetime.now()}] Preparing evaluation instances", log_outfile)
        tok_eval_ds = eval_ds.map(nn_lnk_model.encode_batch, batched=True, batch_size=10**2, desc=f"Preparing {len(eval_ds)} validation instances")
    else:
        tok_eval_ds = None
    print_stdout_file(f"[{dt.datetime.now()}] Preparation done!", log_outfile)
    return run_nn_training(model=nn_lnk_model,
                           epochs=hyperparam_config[NeuralNetworkLinkerConfigKeys.EPOCHS],
                           tok_train_ds=tok_actual_train_ds,
                           tok_eval_ds=tok_eval_ds,
                           log_outfile=log_outfile,
                           export_dir=export_dir,
                           metric=metric,
                           export_at_end=export_at_end)


def train_nn_e2e(train_ds: Dataset,
                 model_class: Type[NeuralNetworkE2E],
                 start_model: NeuralNetworkE2E = None,
                 model_config: dict[NeuralNetworkE2EConfigKeys, Any] = None,
                 export_dir: str = None,
                 export_at_end: bool = True,
                 cache_dir: str = None,
                 hyperparam_config: dict[NeuralNetworkE2EConfigKeys, Any] = None,
                 eval_ds: Dataset = None,
                 metric: str = "loss",
                 log_outfile: str = None) -> tuple[NeuralNetworkE2E, Trainer, dt.timedelta, dict[str, float]]:
    old_train_size = len(train_ds)
    actual_train_ds = augment_ds(train_ds, model_config[NeuralNetworkE2EConfigKeys.FT_AUGMENT_TECH], hyperparam_config[NeuralNetworkE2EConfigKeys.FT_AUGMENT_EXT], log_outfile)
    if len(actual_train_ds) > old_train_size:
        print_stdout_file(f"New training set size: {old_train_size} -> {len(actual_train_ds)}", log_outfile)
    if start_model:
        nn_e2e_model = start_model
    else:
        # TODO Here we have to figure out a way to pass the right class_weights for Finder and Linker. We would need someone that passes these values, as here we don't have access to the Finder and Linker training sets. Actually, we never cover this part of the code... so it's not urgent
        e2e_config_args = {
            "fnd_clf_hidden_size_1": hyperparam_config[NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_1],
            "fnd_clf_hidden_size_2": hyperparam_config[NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_2],
            "lnk_clf_hidden_size_1": hyperparam_config[NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_1],
            "lnk_clf_hidden_size_2": hyperparam_config[NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_2],
            "loss_fun": model_config[NeuralNetworkE2EConfigKeys.LOSS],
            "class_weights": compute_class_weight(class_weight="balanced", classes=np.unique(actual_train_ds[LABEL_COL]), y=actual_train_ds[LABEL_COL]).tolist(),
            "fnd_one_line_text": model_config[NeuralNetworkE2EConfigKeys.FND_ONE_LINE],
            "lnk_one_line_text": model_config[NeuralNetworkE2EConfigKeys.LNK_ONE_LINE],
            "use_cwe": model_config[NeuralNetworkLinkerConfigKeys.USE_CWE],
            "concat_test_first": model_config[NeuralNetworkE2EConfigKeys.MERGE] == MergeStyle.CONCAT_TEST_DESCR,
            "as_javadoc": model_config[NeuralNetworkE2EConfigKeys.MERGE] == MergeStyle.JAVADOC,
            "merge_embeddings": model_config[NeuralNetworkE2EConfigKeys.MERGE] == MergeStyle.LEARN,
            "suspect_threshold": model_config[NeuralNetworkE2EConfigKeys.SUSPECT_THRESHOLD]
        }
        nn_e2e_model = model_class(
            NeuralNetworkE2EConfig(**{k: v for k, v in e2e_config_args.items() if v is not None}),
            cache_dir=cache_dir
        )
    nn_e2e_model.to(DEVICE)
    print_stdout_file(f"Training on device: {DEVICE}", log_outfile)

    print_stdout_file(f"[{dt.datetime.now()}] Preparing training instances", log_outfile)
    tok_actual_train_ds = actual_train_ds.map(nn_e2e_model.encode_batch, batched=True, batch_size=10**3, desc=f"Preparing {len(actual_train_ds)} training instances")
    if eval_ds:
        print_stdout_file(f"[{dt.datetime.now()}] Preparing evaluation instances", log_outfile)
        tok_eval_ds = eval_ds.map(nn_e2e_model.encode_batch, batched=True, batch_size=10**2, desc=f"Preparing {len(eval_ds)} validation instances")
    else:
        tok_eval_ds = None
    print_stdout_file(f"[{dt.datetime.now()}] Preparation done!", log_outfile)
    return run_nn_training(model=nn_e2e_model,
                           epochs=hyperparam_config[NeuralNetworkE2EConfigKeys.FT_EPOCHS],
                           tok_train_ds=tok_actual_train_ds,
                           tok_eval_ds=tok_eval_ds,
                           log_outfile=log_outfile,
                           export_dir=export_dir,
                           metric=metric,
                           export_at_end=export_at_end)
