import gc
import os
import re
import sys
from json import JSONEncoder
from math import sqrt

import datasets
import numpy as np
import torch
from core.common.constants import TEXT_COL
from evaluate import load
from pandas import DataFrame
from sklearn.metrics import average_precision_score, confusion_matrix
from tqdm import tqdm
from transformers import EvalPrediction, TrainerCallback

AUC_ROC = load("roc_auc")


class NpEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def clear_up():
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()


def print_stdout_file(str_to_print: str, filepath: str = None):
    print(str_to_print, flush=True)
    if filepath is not None and os.path.exists(filepath):
        with open(filepath, "a") as fout:
            print(str_to_print, file=fout, flush=True)


def split_dataset(input_df: DataFrame, ratios: list[float], label_col: str, seed: int) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    if sum(ratios) != 1.0:
        return None, None, None
    if len(ratios) < 1 or len(ratios) > 3:
        return None, None, None
    input_ds = datasets.Dataset.from_pandas(input_df, preserve_index=False).class_encode_column(label_col)
    # At least the training set must be requested!
    if ratios[0] is None or ratios[0] == 0.0:
        return None, None, None
    # Pointlessly complex, should change
    if len([r for r in ratios if r is not None and r > 0.0]) == 3:
        eval_test_ratio = ratios[1] + ratios[2]
        test_ratio = ratios[2] / eval_test_ratio
        train_testeval_ds = input_ds.train_test_split(test_size=eval_test_ratio, stratify_by_column=label_col, seed=seed)
        testeval_ds = train_testeval_ds["test"].train_test_split(test_size=test_ratio, stratify_by_column=label_col, seed=seed)
        return train_testeval_ds["train"], testeval_ds["train"], testeval_ds["test"]
    else:
        # No test set
        if ratios[2] is None or ratios[2] == 0.0:
            train_eval_ds = input_ds.train_test_split(test_size=ratios[1], stratify_by_column=label_col, seed=seed)
            return train_eval_ds["train"], train_eval_ds["test"], None
        # No eval set
        if ratios[1] is None or ratios[1] == 0.0:
            train_test_ds = input_ds.train_test_split(test_size=ratios[2], stratify_by_column=label_col, seed=seed)
            return train_test_ds["train"], None, train_test_ds["test"]
        # Fallback
        return None, None, None


def one_line_text(text: str):
    return re.sub(r"\s+", " ", text)


def compute_grouped_estimations(test_df, pred_col, label_col, col_group, col_other, name):
    nr_element_with_expected = 0
    empty_match = 0
    non_empty_match = 0
    underestimations = 0
    overestimations = 0
    groups = test_df.groupby(col_group)
    for _, g_df in groups:
        g_predictions = g_df[g_df[pred_col] == 1][col_other]
        g_expected = g_df[g_df[label_col] == 1][col_other]
        g_predictions_set = set(g_predictions.unique())
        g_expected_set = set(g_expected.unique())
        if len(g_expected_set) == 0:
            if len(g_predictions_set) == 0:
                empty_match += 1
        else:
            nr_element_with_expected += 1
            if g_predictions_set == g_expected_set:
                non_empty_match += 1
            elif g_predictions_set.issubset(g_expected_set):
                underestimations += 1
            elif g_predictions_set.issuperset(g_expected_set):
                overestimations += 1
    return {
        f"{name}_with_expected": nr_element_with_expected,
        f"{name}_empty_match_rate": float(empty_match) / len(groups),
        f"{name}_perfect_prediction_rate": float(non_empty_match) / nr_element_with_expected,
        f"{name}_underestimation_rate": float(underestimations) / nr_element_with_expected,
        f"{name}_overestimation_rate": float(overestimations) / nr_element_with_expected,
    }


def compute_clf_metrics(tp: float, tn: float, fp: float, fn: float):
    p_actu = tp + fn
    n_actu = tn + fp
    p_pred = tp + fp
    n_pred = tn + fn
    pre = tp / p_pred if p_pred > 0 else float('-inf')
    rec = tp / p_actu if p_actu > 0 else float('-inf')
    spe = tn / n_actu if n_actu > 0 else float('-inf')
    fpr = fp / n_actu if n_actu > 0 else float('-inf')
    fnr = fn / p_actu if p_actu > 0 else float('-inf')
    acc = (tp + tn) / (p_actu + n_actu)
    f1 = 2 * pre * rec / (pre + rec) if pre and rec and pre + rec > 0 else float('-inf')
    mcc_den = sqrt(p_pred * p_actu * n_actu * n_pred)
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den and mcc_den > 0 else float('-inf')
    f05 = (1.25 * tp) / ((1.25 * tp) + fp + (0.25 * fn)) if tp + fp + fn > 0 else float('-inf')
    ir = p_pred / (p_actu + n_actu)
    return {
        "p_actu": p_actu,
        "n_actu": n_actu,
        "p_pred": p_pred,
        "n_pred": n_pred,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pre": pre,
        "rec": rec,
        "spe": spe,
        "fpr": fpr,
        "fnr": fnr,
        "acc": acc,
        "f1": f1,
        "mcc": mcc,
        "f05": f05,
        "ir": ir,
    }


def compute_performance_ir(relevant: set, non_relevant: set, retrieved: set):
    tp = len(relevant & retrieved)
    fp = len(non_relevant & retrieved)
    fn = len(relevant - retrieved)
    tn = len(non_relevant - retrieved)
    return compute_clf_metrics(tp=tp, tn=tn, fp=fp, fn=fn)


def compute_performance_clf(expected, predictions, pred_scores=None):
    tn = fp = fn = tp = None
    tn, fp, fn, tp = map(int, confusion_matrix(expected, predictions, labels=[0, 1]).ravel())
    metrics = compute_clf_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    auc_roc = AUC_ROC.compute(prediction_scores=pred_scores, references=expected)["roc_auc"] if pred_scores is not None and len(set(expected)) == 2 else float('-inf')
    auc_pr = average_precision_score(y_true=expected, y_score=pred_scores).item() if pred_scores is not None and len(set(expected)) == 2 else float('-inf')
    return {
        **metrics,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }


def compute_eval_metrics(pred: EvalPrediction):
    expected_labels = pred.label_ids
    actual_labels = pred.predictions.argmax(-1)
    probs = torch.nn.functional.softmax(torch.from_numpy(pred.predictions), dim=-1)
    probs_of_class = probs[:, 1]
    # max_probs, _ = probs.max(-1)
    return compute_performance_clf(expected_labels, actual_labels, probs_of_class)


def check_num_unk_tokens(tokenized_ds: datasets.Dataset, model):
    tot_tokens = 0
    tot_unks = 0
    for train_inst in tqdm(tokenized_ds, total=len(tokenized_ds)):
        tokens = model.tokenizer.convert_ids_to_tokens(train_inst["input_ids"])
        tot_tokens += len(train_inst["input_ids"])
        unk_token_count = tokens.count(model.tokenizer.unk_token)
        if unk_token_count > 0:
            print(f'{train_inst[TEXT_COL]}\nhas {unk_token_count} UNKs out of {len(train_inst["input_ids"])} ({unk_token_count/len(train_inst["input_ids"])})')
            unk_token_positions = [i for i, tok in enumerate(tokens) if tok == model.tokenizer.unk_token]
            for utp in unk_token_positions:
                print(f"Around UNK at position {utp}:")
                if utp > 0:
                    print(f'- BEFORE: {tokens[utp - 1]}')
                if utp < len(train_inst["input_ids"]):
                    print(f'- AFTER: {tokens[utp + 1]}')
            print()
            # print(model.tokenizer.convert_ids_to_tokens(train_inst[TEXT_COL]))
        tot_unks += unk_token_count
    print(f"UNK Token Ratio: {tot_unks} / {tot_tokens} = {tot_unks / tot_tokens}")


def check_long_inputs(input_ds: datasets.Dataset, model, encode_fn, one_line: bool):
    tokenized_ds = input_ds.map(encode_fn, fn_kwargs={
        "tokenizer": model.tokenizer,
        "input_length": model.tokenizer.model_max_length,
        "truncate": False,
        "one_line": one_line
    }, batched=True, batch_size=10**3, desc=f"Preparing {len(input_ds)} training instances")
    long_examples = [ex for ex in tokenized_ds["input_ids"] if len(ex) > model.tokenizer.model_max_length]
    print(f"Long Sequences in Train: {len(long_examples)} / {len(input_ds)} ({len(long_examples) / len(input_ds)})")


class LoggingCallback(TrainerCallback):
    def __init__(self, log_file: str) -> None:
        super().__init__()
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.log_file is not None and os.path.exists(self.log_file):
            with open(self.log_file, "a") as fout:
                print(logs, file=fout, flush=True)
        sys.stdout.flush()
