import datetime as dt
import os
from typing import Any, Type, Union

from core.common import global_vars

if global_vars.MUST_LOAD_UNSLOTH:
    try:
        from unsloth import FastLanguageModel
        UNSLOTH_LOADED = True
    except:
        UNSLOTH_LOADED = False
else:
    UNSLOTH_LOADED = False

import datasets
import numpy as np
import torch
from core.common.constants import (ATTENTION_MASK, DEVICE, INPUT_IDS,
                                   LABEL_COL, TEXT_2_1_COL, TEXT_COL,
                                   E2ETrainingType)
from core.common.utils_training import (compute_performance_clf,
                                        print_stdout_file)
from core.modeling.modeling_lm_common import has_1_in_response
from core.modeling.modeling_lm_e2e import (LanguageModelE2E,
                                           LanguageModelE2EConfig,
                                           LanguageModelE2EConfigKeys)
from core.modeling.modeling_lm_fnd import (LanguageModelFinder,
                                           LanguageModelFinderConfigKeys)
from core.modeling.modeling_lm_lnk import (LanguageModelLinker,
                                           LanguageModelLinkerConfigKeys)
from torch.utils.data import DataLoader
from tqdm import tqdm
from train.evaluation.evaluate_common import (evaluate_e2e,
                                              train_and_test_single)
from train.training.train_lm import train_lm_e2e, train_lm_fnd, train_lm_lnk
from transformers import PreTrainedModel, Trainer


def test_lm_model(lm_model: Union[LanguageModelFinder, LanguageModelLinker],
                  test_ds: datasets.Dataset,
                  log_outfile: str,
                  #  trainer: Trainer = None
                  ) -> dict:
    print_stdout_file(f"[{dt.datetime.now()}] Preparing test instances", log_outfile)
    prep_test_ds = test_ds.map(lm_model.prepare_batch_eval,
                               batched=True,
                               batch_size=10**2,
                               desc=f"Preparing {len(test_ds)} test instances",
                               remove_columns=[c for c in test_ds.column_names if c in ['file', 'class', 'method', 'url', 'fixes', TEXT_COL, TEXT_2_1_COL]])
    print_stdout_file(f"[{dt.datetime.now()}] Preparation done!", log_outfile)

    if lm_model.config.unsloth_training and UNSLOTH_LOADED:
        # Apparently, it should enable faster inference
        FastLanguageModel.for_inference(lm_model.lm_model)
        print("Padding side:")
        print(lm_model.tokenizer.padding_side)
        # We force padding to the left!
        lm_model.tokenizer.padding_side = "left"

    # TODO LLM - DEBUG
    #  prep_test_ds = prep_test_ds.select(range(12))

    print_stdout_file(f"[{dt.datetime.now()}] Making predictions...", log_outfile)
    test_responses = []
    lm_model.to(DEVICE)
    lm_model.eval()
    with torch.no_grad():
        for test_batch in tqdm(DataLoader(prep_test_ds.with_format("torch"), batch_size=8)):
            input_ids = test_batch[INPUT_IDS].to(lm_model.lm_model.device)
            attention_mask = test_batch[ATTENTION_MASK].to(lm_model.lm_model.device)
            gen_output = lm_model.do_generate(input_ids=input_ids, attention_mask=attention_mask).detach().cpu().numpy()
            del input_ids, attention_mask
            test_responses.extend(lm_model.extract_responses(gen_output))
    predicted_classes = [int(has_1_in_response(gt)) for gt in test_responses]
    label_ids = np.asarray(prep_test_ds[LABEL_COL])
    test_predictions = np.asarray(predicted_classes)
    test_perf = compute_performance_clf(label_ids, test_predictions)
    # NOTE LLM - We use the same strategy of get_witnessing_score()
    probs_of_class = [1.0] * len(test_predictions)
    print_stdout_file(f"[{dt.datetime.now()}] Testing finished!", log_outfile)
    return {
        "test_performance": {k: v for k, v in test_perf.items()},
        "test_predictions": test_predictions,
        "test_responses": test_responses,
        "test_probabilities": probs_of_class,
    }


def evaluate_lm_fnd(train_ds: datasets.Dataset,
                    test_ds: datasets.Dataset,
                    model_config: dict[LanguageModelFinderConfigKeys, Any],
                    log_outfile: str,
                    model_class: Type[LanguageModelFinder],
                    export_dir: str,
                    cache_dir: str = None,
                    hyperparam_space: dict[LanguageModelFinderConfigKeys, list[Any]] = None,
                    eval_ds: datasets.Dataset = None,
                    metric: str = "loss") -> tuple[dict, PreTrainedModel]:
    return train_and_test_single(train_ds,
                                 test_ds,
                                 model_config,
                                 log_outfile,
                                 train_fn=train_lm_fnd,
                                 test_fn=test_lm_model,
                                 model_class=model_class,
                                 export_dir=export_dir,
                                 cache_dir=cache_dir,
                                 hp_space=hyperparam_space,
                                 eval_ds=eval_ds,
                                 metric=metric)


def evaluate_lm_lnk(train_ds: datasets.Dataset,
                    test_ds: datasets.Dataset,
                    model_config: dict[LanguageModelLinkerConfigKeys, Any],
                    log_outfile: str,
                    model_class: Type[LanguageModelLinker],
                    export_dir: str,
                    cache_dir: str = None,
                    hyperparam_space: dict[LanguageModelLinkerConfigKeys, list[Any]] = None,
                    eval_ds: datasets.Dataset = None,
                    metric: str = "loss") -> tuple[dict, PreTrainedModel]:
    return train_and_test_single(train_ds,
                                 test_ds,
                                 model_config,
                                 log_outfile,
                                 train_fn=train_lm_lnk,
                                 test_fn=test_lm_model,
                                 model_class=model_class,
                                 export_dir=export_dir,
                                 cache_dir=cache_dir,
                                 hp_space=hyperparam_space,
                                 eval_ds=eval_ds,
                                 metric=metric)


def e2e_to_lnk_config(model_config: dict[LanguageModelE2EConfigKeys, Any],
                      hyperparam_config: dict[LanguageModelE2EConfigKeys, Any]) -> tuple[dict, dict]:
    lnk_config = {
        LanguageModelLinkerConfigKeys.UNSLOTH_TRAINING: model_config[LanguageModelE2EConfigKeys.UNSLOTH_TRAINING],
        LanguageModelLinkerConfigKeys.USE_CWE: model_config[LanguageModelE2EConfigKeys.USE_CWE],
        LanguageModelLinkerConfigKeys.AUGMENT_TECH: model_config[LanguageModelE2EConfigKeys.LNK_AUGMENT_TECH],
    }
    lnk_hp = {
        LanguageModelLinkerConfigKeys.AUGMENT_EXT: hyperparam_config[LanguageModelE2EConfigKeys.LNK_AUGMENT_EXT],
        LanguageModelLinkerConfigKeys.EPOCHS: hyperparam_config[LanguageModelE2EConfigKeys.LNK_EPOCHS],
    }
    return lnk_config, lnk_hp


def build_lm_e2e(
    e2e_model_config: dict[LanguageModelE2EConfigKeys, Any],
    e2e_model_class: Type[LanguageModelE2E],
    export_dir: str,
    log_outfile: str,
    fnd_train_ds: datasets.Dataset = None,
    fnd_eval_ds: datasets.Dataset = None,
    lnk_train_ds: datasets.Dataset = None,
    lnk_eval_ds: datasets.Dataset = None,
    e2e_train_ds: datasets.Dataset = None,
    e2e_hyperparam_config: dict[LanguageModelE2EConfigKeys, Any] = None,
    e2e_eval_ds: datasets.Dataset = None,
    metric: str = "loss",
    cache_dir: str = None,
    export_at_end: bool = True
) -> tuple[PreTrainedModel, Trainer, dict[str, dt.timedelta], dict]:
    e2e_config_args = {
        "unsloth_training": e2e_model_config[LanguageModelE2EConfigKeys.UNSLOTH_TRAINING],
        "use_cwe": e2e_model_config[LanguageModelE2EConfigKeys.USE_CWE],
        "archi_style": e2e_model_config[LanguageModelE2EConfigKeys.ARCHI_STYLE],
    }
    e2e_config = LanguageModelE2EConfig(**{k: v for k, v in e2e_config_args.items() if v is not None})
    lnk_duration = dt.timedelta(0)
    e2e_duration = dt.timedelta(0)

    if e2e_model_config[LanguageModelE2EConfigKeys.TRAIN_TYPE] in [E2ETrainingType.PRETRAIN_ONLY, E2ETrainingType.PRETRAIN_FINETUNE]:
        print_stdout_file(f"[{dt.datetime.now()}] Going to train the Linker module", log_outfile)
        lnk_config, lnk_hp = e2e_to_lnk_config(e2e_model_config, e2e_hyperparam_config)
        lnk_export_dir = os.path.join(export_dir, "lnk")
        lnk_model, _, lnk_duration, _ = train_lm_lnk(train_ds=lnk_train_ds,
                                                     model_class=e2e_model_class.LNK_MODEL_CLASS,
                                                     model_config=lnk_config,
                                                     log_outfile=log_outfile,
                                                     export_dir=lnk_export_dir,
                                                     export_at_end=False,
                                                     cache_dir=cache_dir,
                                                     hyperparam_config=lnk_hp,
                                                     eval_ds=lnk_eval_ds,
                                                     metric=metric
                                                     )
        e2e_model = e2e_model_class(e2e_config, lnk_model=lnk_model, cache_dir=cache_dir)
    else:
        print_stdout_file(f"[{dt.datetime.now()}] Loading Finder and/or Linker with default weights", log_outfile)
        e2e_model = e2e_model_class(e2e_config, cache_dir=cache_dir)

    if e2e_model_config[LanguageModelE2EConfigKeys.TRAIN_TYPE] in [E2ETrainingType.FINETUNE_ONLY, E2ETrainingType.PRETRAIN_FINETUNE]:
        print_stdout_file(f"[{dt.datetime.now()}] Going to train the End-to-End module", log_outfile)
        e2e_export_dir = os.path.join(export_dir, "e2e")
        e2e_model, e2e_trainer, e2e_duration, e2e_dev_results = train_lm_e2e(train_ds=e2e_train_ds,
                                                                             model_class=e2e_model_class,
                                                                             start_model=e2e_model,
                                                                             model_config=e2e_model_config,
                                                                             log_outfile=log_outfile,
                                                                             export_dir=e2e_export_dir,
                                                                             export_at_end=export_at_end,
                                                                             cache_dir=cache_dir,
                                                                             hyperparam_config=e2e_hyperparam_config,
                                                                             eval_ds=e2e_eval_ds,
                                                                             metric=metric)
    else:
        e2e_trainer = None
        e2e_dev_results = {}
    return e2e_model, e2e_trainer, {"lnk": lnk_duration, "e2e": e2e_duration}, e2e_dev_results


def evaluate_lm_e2e(
    model_config: dict[LanguageModelE2EConfigKeys, Any],
    model_class: Type[LanguageModelE2E],
    export_dir: str,
    log_outfile: str,
    e2e_test_ds: datasets.Dataset,
    fnd_train_ds: datasets.Dataset = None,
    fnd_eval_ds: datasets.Dataset = None,
    lnk_train_ds: datasets.Dataset = None,
    lnk_eval_ds: datasets.Dataset = None,
    e2e_train_ds: datasets.Dataset = None,
    hyperparam_space: dict[LanguageModelE2EConfigKeys, list[Any]] = None,
    e2e_eval_ds: datasets.Dataset = None,
    metric: str = "loss",
    cache_dir: str = None,
) -> tuple[dict, PreTrainedModel]:
    return evaluate_e2e(
        e2e_model_build_fn=build_lm_e2e,
        e2e_model_test_fn=test_lm_model,
        model_config=model_config,
        model_class=model_class,
        export_dir=export_dir,
        log_outfile=log_outfile,
        e2e_test_ds=e2e_test_ds,
        fnd_train_ds=fnd_train_ds,
        fnd_eval_ds=fnd_eval_ds,
        lnk_train_ds=lnk_train_ds,
        lnk_eval_ds=lnk_eval_ds,
        e2e_train_ds=e2e_train_ds,
        hyperparam_space=hyperparam_space,
        e2e_eval_ds=e2e_eval_ds,
        metric=metric,
        cache_dir=cache_dir
    )
