from typing import Any

import datasets
import numpy as np
from tqdm import tqdm

from vuteco.core.common.constants import LABEL_COL, TEXT_COL
from vuteco.core.common.utils_training import (compute_performance_clf,
                                               print_stdout_file)
from vuteco.core.modeling.modeling_grep_fnd import (GrepFinder,
                                                    GrepFinderConfigKeys)


def eval_grep_fnd(train_ds: datasets.Dataset,
                  test_ds: datasets.Dataset,
                  model_config: dict[GrepFinderConfigKeys, Any],
                  log_outfile: str,
                  hyperparam_space: dict[GrepFinderConfigKeys, list[Any]] = None,
                  eval_ds: datasets.Dataset = None) -> tuple[dict, GrepFinder]:
    grep_fnd_model = GrepFinder(**model_config)
    if eval_ds is not None:
        unflagged_sec_tests = [i[TEXT_COL] for i in eval_ds if i[LABEL_COL] == 1]
        pat_errors = {}
        pat_hits = {}
        for inst in tqdm(eval_ds, desc="Predictions on the Validation Set"):
            flagged, patterns_matched = grep_fnd_model.is_security_test(inst[TEXT_COL], return_pattern=True)
            for pat in patterns_matched:
                try:
                    unflagged_sec_tests.remove(inst[TEXT_COL])
                except:
                    pass
                if pat not in pat_errors:
                    pat_errors[pat] = 0
                if pat not in pat_hits:
                    pat_hits[pat] = 0
                pat_errors[pat] += 0 if inst[LABEL_COL] else 1
                pat_hits[pat] += 1 if inst[LABEL_COL] else 0
        # print(f"False Negatives: {len(unflagged_sec_tests)}")
        # for st in unflagged_sec_tests:
        #    print(st)
        #    print("Press ENTER to continue...")
        #    input()
        print_stdout_file("Errors made by the patterns:", log_outfile)
        print_stdout_file(sorted(pat_errors.items(), key=lambda x: x[1], reverse=True), log_outfile)
        print_stdout_file("", log_outfile)
        print_stdout_file("Hit made by the patterns:", log_outfile)
        print_stdout_file(sorted(pat_hits.items(), key=lambda x: x[1], reverse=True), log_outfile)
    predictions = [int(grep_fnd_model.is_security_test(inst[TEXT_COL], return_pattern=True)[0]) for inst in tqdm(test_ds, desc="Predictions on the Test Set")]
    test_results = compute_performance_clf(test_ds[LABEL_COL], predictions)
    return {
        "model_config": model_config,
        "test_results": {
            "test_performance": {k: None if np.isnan(v) else v for k, v in test_results.items()},
            "test_predictions": predictions
        }
    }, grep_fnd_model
