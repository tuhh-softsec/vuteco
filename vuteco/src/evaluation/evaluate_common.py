import datetime as dt
import itertools
import json
from collections.abc import Callable, Iterable
from typing import Any, Type, Union

import datasets
from common.utils_training import clear_up, print_stdout_file
from modeling.modeling_lm_fnd import LanguageModelFinderConfigKeys
from modeling.modeling_lm_lnk import LanguageModelLinkerConfigKeys
from modeling.modeling_nn_fnd import NeuralNetworkFinderConfigKeys
from modeling.modeling_nn_lnk import NeuralNetworkLinkerConfigKeys
from transformers import PreTrainedModel


def config_to_str(ec: dict) -> str:
    parts = []
    for v in ec.values():
        if isinstance(v, Iterable) and not isinstance(v, str):
            parts.append("-".join(str(e) for e in v))
        else:
            parts.append(str(v))
    return "_".join(parts)


def str_to_config(ec_str: str, keys: list[str]) -> dict:
    splits = ec_str.split("_")
    return {ck.value: splits[idx] for idx, ck in enumerate(keys)}


def product_dict(inp: dict):
    return (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))


def get_combos(inp: dict):
    return list(product_dict(inp))


def is_results_better(results_1: dict, results_2: dict, metric: str = "loss"):
    if results_1 is None:
        return False
    if results_2 is None:
        return True
    if metric in ["loss", "fpr", "fnr"]:
        if results_1[f'eval_{metric}'] < results_2[f'eval_{metric}']:
            return True
        elif results_1[f'eval_loss'] < results_2[f'eval_loss']:
            return True
        else:
            return False
    else:
        if results_1[f'eval_{metric}'] > results_2[f'eval_{metric}']:
            return True
        elif results_1[f'eval_loss'] < results_2[f'eval_loss']:
            return True
        else:
            return False


# def get_best_config(hp_search_log: list[tuple], metric: str):
#     if metric in ["loss", "fpr", "fnr"]:
#         _, best_hp_config, _ = min(hp_search_log, key=lambda x: (
#             x[0][f'eval_{metric}'] if x[0] is not None and x[0][f'eval_{metric}'] is not None else float('inf'),
#             x[0]['eval_loss'] if x[0] is not None and x[0][f'eval_loss'] is not None else float('inf')
#         ))
#     else:
#         _, best_hp_config, _ = max(hp_search_log, key=lambda x: (
#             x[0][f'eval_{metric}'] if x[0] is not None and x[0][f'eval_{metric}'] is not None else float('-inf'),
#             -x[0]['eval_loss'] if x[0] is not None and x[0][f'eval_loss'] is not None else float('-inf')
#         ))
#     return best_hp_config


def train_and_test_single(train_ds: datasets.Dataset,
                          test_ds: datasets.Dataset,
                          model_config: dict[Union[NeuralNetworkFinderConfigKeys, NeuralNetworkLinkerConfigKeys, LanguageModelFinderConfigKeys, LanguageModelLinkerConfigKeys], Any],
                          log_outfile: str,
                          train_fn: Callable,
                          test_fn: Callable,
                          model_class: Type[PreTrainedModel],
                          export_dir: str = None,
                          cache_dir: str = None,
                          hp_space: dict[Union[NeuralNetworkFinderConfigKeys, NeuralNetworkLinkerConfigKeys,
                                               LanguageModelFinderConfigKeys, LanguageModelLinkerConfigKeys], list[Any]] = None,
                          eval_ds: datasets.Dataset = None,
                          metric: str = "loss") -> tuple[dict, PreTrainedModel]:
    hp_search_log = []
    best_so_far = (None, None, None, None)
    for hp_config in get_combos(hp_space):
        print_stdout_file(f"\nTraining model with the following hyperparameters '{config_to_str(hp_config)}'", log_outfile)
        print_stdout_file(json.dumps(hp_config, indent=2), log_outfile)
        model, unneeded_trainer, train_duration, eval_results = train_fn(train_ds,
                                                                         model_class,
                                                                         model_config=model_config,
                                                                         export_dir=export_dir,
                                                                         export_at_end=False,
                                                                         cache_dir=cache_dir,
                                                                         hyperparam_config=hp_config,
                                                                         eval_ds=eval_ds,
                                                                         metric=metric,
                                                                         log_outfile=log_outfile)
        del unneeded_trainer
        clear_up()
        hp_search_log.append((eval_results, hp_config, train_duration))
        if best_so_far == (None, None, None, None) or is_results_better(eval_results, best_so_far[0], metric):
            best_so_far = (eval_results, hp_config, model, train_duration)
            # NOTE To save memory, we move the model to CPU, as we don't need it the GPU right now. If still we have memory errors, presist on the disk.
            model.to("cpu")
        else:
            del model

    _, best_hp_config, best_model, best_duration = best_so_far
    print_stdout_file(f"\nModel with the optimal hyperparameters '{config_to_str(best_hp_config)}'", log_outfile)
    # best_hp_config = get_best_config(hp_search_log, metric)
    # print_stdout_file(f"\nTraining model with the optimal hyperparameters '{config_to_str(best_hp_config)}'", log_outfile)
    # print_stdout_file(json.dumps(hp_config, indent=2), log_outfile)
    # final_model, trainer, final_duration, _ = train_fn(train_ds,
    #                                              model_class,
    #                                              model_config=model_config,
    #                                              export_dir=export_dir,
    #                                              cache_dir=cache_dir,
    #                                              hyperparam_config=best_hp_config,
    #                                              eval_ds=eval_ds,
    #                                              metric=metric,
    #                                              log_outfile=log_outfile)
    test_results = test_fn(best_model,
                           test_ds,
                           log_outfile,
                           # trainer=best_trainer
                           )
    return {
        "model_config": model_config,
        "hyperparams": best_hp_config,
        "train_time": str(best_duration),
        "test_results": test_results,
        "optimized_metric": metric,
        "hyperparam_space": hp_space,
        "hyperparam_results": [{"hyperparams": hp, "train_time": str(td), **dr} for dr, hp, td in hp_search_log],
    }, best_model


def evaluate_e2e(
    e2e_model_build_fn: Callable,
    e2e_model_test_fn: Callable,
    model_config: dict[str, Any],
    model_class: Type[PreTrainedModel],
    export_dir: str,
    log_outfile: str,
    e2e_test_ds: datasets.Dataset,
    fnd_train_ds: datasets.Dataset = None,
    fnd_eval_ds: datasets.Dataset = None,
    lnk_train_ds: datasets.Dataset = None,
    lnk_eval_ds: datasets.Dataset = None,
    e2e_train_ds: datasets.Dataset = None,
    hyperparam_space: dict[str, list[Any]] = None,
    e2e_eval_ds: datasets.Dataset = None,
    metric: str = "loss",
    cache_dir: str = None,
) -> tuple[dict, PreTrainedModel]:
    hp_search_log = []
    best_so_far = (None, None, None, None)
    for hp_config in get_combos(hyperparam_space):
        print_stdout_file(f"\nTraining model with the following hyperparameters '{config_to_str(hp_config)}'", log_outfile)
        print_stdout_file(json.dumps(hp_config, indent=2), log_outfile)
        model, _, durations, e2e_eval_results = e2e_model_build_fn(model_config,
                                                                   model_class,
                                                                   export_dir,
                                                                   log_outfile,
                                                                   fnd_train_ds=fnd_train_ds,
                                                                   fnd_eval_ds=fnd_eval_ds,
                                                                   lnk_train_ds=lnk_train_ds,
                                                                   lnk_eval_ds=lnk_eval_ds,
                                                                   e2e_train_ds=e2e_train_ds,
                                                                   e2e_hyperparam_config=hp_config,
                                                                   e2e_eval_ds=e2e_eval_ds,
                                                                   metric=metric,
                                                                   cache_dir=cache_dir,
                                                                   export_at_end=False)
        hp_search_log.append((e2e_eval_results, hp_config, sum(durations.values(), dt.timedelta())))
        if best_so_far == (None, None, None, None) or is_results_better(e2e_eval_results, best_so_far[0], metric):
            best_so_far = (e2e_eval_results, hp_config, model, durations)
            # NOTE To save memory, we move the model to CPU, as we don't need it the GPU right now. If still we have memory errors, presist on the disk.
            model.to("cpu")
        else:
            del model

    _, best_hp_config, best_model, best_duration = best_so_far
    print_stdout_file(f"\nModel with the optimal hyperparameters '{config_to_str(best_hp_config)}'", log_outfile)
    test_results = e2e_model_test_fn(best_model,
                                     e2e_test_ds,
                                     log_outfile,
                                     #  trainer=best_trainer
                                     )
    return {
        "model_config": model_config,
        "hyperparams": best_hp_config,
        "train_time": {
            "total": str(sum(best_duration.values(), dt.timedelta())),
            **{k: str(v) for k, v in best_duration.items()}
        },
        "test_results": test_results,
        "optimized_metric": metric,
        "hyperparam_space": hyperparam_space,
        "hyperparam_results": [{"hyperparams": hp, "train_time": str(td), **dr} for dr, hp, td in hp_search_log],
    }, best_model
