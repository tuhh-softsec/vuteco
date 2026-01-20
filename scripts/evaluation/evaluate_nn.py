import datetime as dt
import os
from typing import Any, Type

import datasets
import numpy as np
import torch
from common.constants import (DEVICE, LABEL_COL, TEXT_2_1_COL,
                              E2EArchitectureStyle, E2ETrainingType,
                              MergeStyle)
from common.utils_training import compute_performance_clf, print_stdout_file
from evaluation.evaluate_common import evaluate_e2e, train_and_test_single
from modeling.modeling_nn_e2e import (NeuralNetworkE2E, NeuralNetworkE2EConfig,
                                      NeuralNetworkE2EConfigKeys)
from modeling.modeling_nn_fnd import (NeuralNetworkFinder,
                                      NeuralNetworkFinderConfigKeys)
from modeling.modeling_nn_lnk import (NeuralNetworkLinker,
                                      NeuralNetworkLinkerConfigKeys)
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.train_nn import train_nn_e2e, train_nn_fnd, train_nn_lnk
from transformers import PreTrainedModel, Trainer


def test_nn_model(nn_model: PreTrainedModel,
                  test_ds: datasets.Dataset,
                  log_outfile: str,
                  # trainer: Trainer = None
                  ) -> dict:
    print_stdout_file(f"[{dt.datetime.now()}] Preparing test instances", log_outfile)
    # NOTE Since datasets.map() keeps the original columns (!), we are forced to explicitly drop the columns we don't need and that might be null. Why? Because when we use torch.DataLoader (see below) an error occurs when there's a null value while making the Tensors. trainer.predict() is robust to this scenario.
    tok_test_ds = test_ds.map(nn_model.encode_batch,
                              batched=True,
                              batch_size=10**2,
                              desc=f"Preparing {len(test_ds)} test instances",
                              remove_columns=[c for c in test_ds.column_names if c in ['file', 'class', 'method', 'url', 'fixes', TEXT_2_1_COL]])
    print_stdout_file(f"[{dt.datetime.now()}] Preparation done!", log_outfile)

    print_stdout_file(f"[{dt.datetime.now()}] Making predictions...", log_outfile)
    # if trainer is not None:
    #     test_predictions, test_labels, test_perf = trainer.predict(tok_test_ds)
    #     predicted_classes = test_predictions.argmax(-1)
    #     probs = torch.nn.functional.softmax(torch.from_numpy(test_predictions), dim=-1)
    #     probs_of_class = probs[:, 1].detach().cpu().numpy()
    #  else:
    all_logits = []
    nn_model.to(DEVICE)
    nn_model.eval()
    with torch.no_grad():
        for test_batch in tqdm(DataLoader(tok_test_ds.with_format("torch"), batch_size=8)):
            batch_as_dict = {n: test_batch[n].to(DEVICE) if n in test_batch else None for n in nn_model.tokenizer.model_input_names}
            batch_outputs = nn_model(**batch_as_dict)
            batch_logits = batch_outputs.logits
            all_logits.append(batch_logits)
    test_logits = torch.cat(all_logits, dim=0)
    predicted_classes = test_logits.argmax(-1).detach().cpu().numpy()
    probs = torch.nn.functional.softmax(test_logits, dim=-1)
    probs_of_class = probs[:, 1].detach().cpu().numpy()
    test_perf = compute_performance_clf(tok_test_ds[LABEL_COL], predicted_classes, pred_scores=probs_of_class)
    # print_stdout_file(f"Test Logits -> {test_logits}", session_outfile)
    # print_stdout_file(f"Test Probs -> {torch.nn.functional.softmax(torch.from_numpy(test_logits), dim=-1).numpy()}", session_outfile)
    # print_stdout_file(f"Test Preds -> {predictions}", session_outfile)
    # print_stdout_file(f"Test Labels -> {test_labels}", session_outfile)
    print_stdout_file(f"[{dt.datetime.now()}] Testing finished!", log_outfile)
    return {
        "test_performance": {k: v for k, v in test_perf.items()},
        "test_predictions": predicted_classes,
        "test_probabilities": probs_of_class
    }


def evaluate_nn_fnd(train_ds: datasets.Dataset,
                    test_ds: datasets.Dataset,
                    model_config: dict[NeuralNetworkFinderConfigKeys, Any],
                    log_outfile: str,
                    model_class: Type[NeuralNetworkFinder],
                    export_dir: str,
                    cache_dir: str = None,
                    hyperparam_space: dict[NeuralNetworkFinderConfigKeys, list[Any]] = None,
                    eval_ds: datasets.Dataset = None,
                    metric: str = "loss") -> tuple[dict, PreTrainedModel]:
    return train_and_test_single(train_ds,
                                 test_ds,
                                 model_config,
                                 log_outfile,
                                 train_fn=train_nn_fnd,
                                 test_fn=test_nn_model,
                                 model_class=model_class,
                                 export_dir=export_dir,
                                 cache_dir=cache_dir,
                                 hp_space=hyperparam_space,
                                 eval_ds=eval_ds,
                                 metric=metric)


def evaluate_nn_lnk(train_ds: datasets.Dataset,
                    test_ds: datasets.Dataset,
                    model_config: dict[NeuralNetworkLinkerConfigKeys, Any],
                    log_outfile: str,
                    model_class: Type[NeuralNetworkLinker],
                    export_dir: str,
                    cache_dir: str = None,
                    hyperparam_space: dict[NeuralNetworkLinkerConfigKeys, list[Any]] = None,
                    eval_ds: datasets.Dataset = None,
                    metric: str = "loss") -> tuple[dict, PreTrainedModel]:
    return train_and_test_single(train_ds,
                                 test_ds,
                                 model_config,
                                 log_outfile,
                                 train_fn=train_nn_lnk,
                                 test_fn=test_nn_model,
                                 model_class=model_class,
                                 export_dir=export_dir,
                                 cache_dir=cache_dir,
                                 hp_space=hyperparam_space,
                                 eval_ds=eval_ds,
                                 metric=metric)


######################
# E2E Model Business #
######################

def e2e_to_fnd_config(model_config: dict[NeuralNetworkE2EConfigKeys, Any],
                      hyperparam_config: dict[NeuralNetworkE2EConfigKeys, Any]) -> tuple[dict, dict]:
    fnd_config = {
        NeuralNetworkFinderConfigKeys.AUGMENT_TECH: model_config[NeuralNetworkE2EConfigKeys.FND_AUGMENT_TECH],
        NeuralNetworkFinderConfigKeys.LOSS: model_config[NeuralNetworkE2EConfigKeys.LOSS],
        NeuralNetworkFinderConfigKeys.ONE_LINE: model_config[NeuralNetworkE2EConfigKeys.FND_ONE_LINE]
    }
    fnd_hp = {
        NeuralNetworkFinderConfigKeys.AUGMENT_EXT: hyperparam_config[NeuralNetworkE2EConfigKeys.FND_AUGMENT_EXT],
        NeuralNetworkFinderConfigKeys.HIDDEN_SIZE_1: hyperparam_config[NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_1],
        NeuralNetworkFinderConfigKeys.HIDDEN_SIZE_2: hyperparam_config[NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_2],
        NeuralNetworkFinderConfigKeys.EPOCHS: hyperparam_config[NeuralNetworkE2EConfigKeys.FND_EPOCHS],
    }
    return fnd_config, fnd_hp


def e2e_to_lnk_config(model_config: dict[NeuralNetworkE2EConfigKeys, Any],
                      hyperparam_config: dict[NeuralNetworkE2EConfigKeys, Any]) -> tuple[dict, dict]:
    lnk_config = {
        NeuralNetworkLinkerConfigKeys.MERGE: model_config[NeuralNetworkE2EConfigKeys.MERGE],
        NeuralNetworkLinkerConfigKeys.USE_CWE: model_config[NeuralNetworkE2EConfigKeys.USE_CWE],
        NeuralNetworkLinkerConfigKeys.AUGMENT_TECH: model_config[NeuralNetworkE2EConfigKeys.LNK_AUGMENT_TECH],
        NeuralNetworkLinkerConfigKeys.LOSS: model_config[NeuralNetworkE2EConfigKeys.LOSS],
        NeuralNetworkLinkerConfigKeys.ONE_LINE: model_config[NeuralNetworkE2EConfigKeys.LNK_ONE_LINE]
    }
    lnk_hp = {
        NeuralNetworkLinkerConfigKeys.AUGMENT_EXT: hyperparam_config[NeuralNetworkE2EConfigKeys.LNK_AUGMENT_EXT],
        NeuralNetworkLinkerConfigKeys.HIDDEN_SIZE_1: hyperparam_config[NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_1],
        NeuralNetworkLinkerConfigKeys.HIDDEN_SIZE_2: hyperparam_config[NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_2],
        NeuralNetworkLinkerConfigKeys.EPOCHS: hyperparam_config[NeuralNetworkE2EConfigKeys.LNK_EPOCHS],
    }
    return lnk_config, lnk_hp


def build_nn_e2e(
    e2e_model_config: dict[NeuralNetworkE2EConfigKeys, Any],
    e2e_model_class: Type[NeuralNetworkE2E],
    export_dir: str,
    log_outfile: str,
    fnd_train_ds: datasets.Dataset = None,
    fnd_eval_ds: datasets.Dataset = None,
    lnk_train_ds: datasets.Dataset = None,
    lnk_eval_ds: datasets.Dataset = None,
    e2e_train_ds: datasets.Dataset = None,
    e2e_hyperparam_config: dict[NeuralNetworkE2EConfigKeys, Any] = None,
    e2e_eval_ds: datasets.Dataset = None,
    metric: str = "loss",
    cache_dir: str = None,
    export_at_end: bool = True
) -> tuple[PreTrainedModel, Trainer, dict[str, dt.timedelta], dict]:
    e2e_config_args = {
        "fnd_clf_hidden_size_1": e2e_hyperparam_config[NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_1],
        "fnd_clf_hidden_size_2": e2e_hyperparam_config[NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_2],
        "fnd_loss_fun": e2e_model_config[NeuralNetworkE2EConfigKeys.FND_LOSS],
        "fnd_class_weights": compute_class_weight(class_weight="balanced", classes=np.unique(fnd_train_ds[LABEL_COL]), y=fnd_train_ds[LABEL_COL]).tolist(),
        "lnk_clf_hidden_size_1": e2e_hyperparam_config[NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_1],
        "lnk_clf_hidden_size_2": e2e_hyperparam_config[NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_2],
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
    e2e_config = NeuralNetworkE2EConfig(**{k: v for k, v in e2e_config_args.items() if v is not None})
    fnd_duration = dt.timedelta(0)
    lnk_duration = dt.timedelta(0)
    e2e_duration = dt.timedelta(0)

    if e2e_model_config[NeuralNetworkE2EConfigKeys.TRAIN_TYPE] in [E2ETrainingType.PRETRAIN_ONLY, E2ETrainingType.PRETRAIN_FINETUNE]:
        if e2e_model_config[NeuralNetworkE2EConfigKeys.ARCHI_STYLE] in [E2EArchitectureStyle.LINKER_ONLY]:
            fnd_model = None
        else:
            print_stdout_file(f"[{dt.datetime.now()}] Going to train the Finder module", log_outfile)
            fnd_config, fnd_hp = e2e_to_fnd_config(e2e_model_config, e2e_hyperparam_config)
            fnd_export_dir = os.path.join(export_dir, "fnd")
            fnd_model, _, fnd_duration, _ = train_nn_fnd(train_ds=fnd_train_ds,
                                                        model_class=e2e_model_class.FND_MODEL_CLASS,
                                                        model_config=fnd_config,
                                                        log_outfile=log_outfile,
                                                        export_dir=fnd_export_dir,
                                                        export_at_end=False,
                                                        cache_dir=cache_dir,
                                                        hyperparam_config=fnd_hp,
                                                        eval_ds=fnd_eval_ds,
                                                        metric=metric
                                                        )
        print_stdout_file(f"[{dt.datetime.now()}] Going to train the Linker module", log_outfile)
        lnk_config, lnk_hp = e2e_to_lnk_config(e2e_model_config, e2e_hyperparam_config)
        lnk_export_dir = os.path.join(export_dir, "lnk")
        lnk_model, _, lnk_duration, lnk_dev_results = train_nn_lnk(train_ds=lnk_train_ds,
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
        e2e_model = e2e_model_class(e2e_config, fnd_model=fnd_model, lnk_model=lnk_model, cache_dir=cache_dir)
    else:
        print_stdout_file(f"[{dt.datetime.now()}] Loading Finder and/or Linker with default weights", log_outfile)
        e2e_model = e2e_model_class(e2e_config, cache_dir=cache_dir)

    if e2e_model_config[NeuralNetworkE2EConfigKeys.TRAIN_TYPE] in [E2ETrainingType.FINETUNE_ONLY, E2ETrainingType.PRETRAIN_FINETUNE]:
        print_stdout_file(f"[{dt.datetime.now()}] Going to train the End-to-End module", log_outfile)
        e2e_export_dir = os.path.join(export_dir, "e2e")
        e2e_model, e2e_trainer, e2e_duration, e2e_dev_results = train_nn_e2e(train_ds=e2e_train_ds,
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
        e2e_dev_results = lnk_dev_results
    return e2e_model, e2e_trainer, {"fnd": fnd_duration, "lnk": lnk_duration, "e2e": e2e_duration}, e2e_dev_results


def evaluate_nn_e2e(
    model_config: dict[NeuralNetworkE2EConfigKeys, Any],
    model_class: Type[NeuralNetworkE2E],
    export_dir: str,
    log_outfile: str,
    e2e_test_ds: datasets.Dataset,
    fnd_train_ds: datasets.Dataset = None,
    fnd_eval_ds: datasets.Dataset = None,
    lnk_train_ds: datasets.Dataset = None,
    lnk_eval_ds: datasets.Dataset = None,
    e2e_train_ds: datasets.Dataset = None,
    hyperparam_space: dict[NeuralNetworkE2EConfigKeys, list[Any]] = None,
    e2e_eval_ds: datasets.Dataset = None,
    metric: str = "loss",
    cache_dir: str = None,
) -> tuple[dict, PreTrainedModel]:
    return evaluate_e2e(
        e2e_model_build_fn=build_nn_e2e,
        e2e_model_test_fn=test_nn_model,
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
