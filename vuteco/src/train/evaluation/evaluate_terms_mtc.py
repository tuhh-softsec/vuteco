from typing import Any

import datasets
import numpy as np
from core.common.constants import LABEL_COL, TEXT_1_COL, TEXT_2_COL
from core.common.utils_training import compute_performance_clf
from core.modeling.modeling_terms_mtc import (TermsMatcher,
                                              TermsMatcherConfigKeys)
from tqdm import tqdm


def eval_terms_mtc(train_ds: datasets.Dataset,
                   test_ds: datasets.Dataset,
                   model_config: dict[TermsMatcherConfigKeys, Any],
                   log_outfile: str,
                   hyperparam_space: dict[TermsMatcherConfigKeys, list[Any]] = None) -> tuple[dict, TermsMatcher]:
    terms_mtc_model = TermsMatcher(**model_config)
    predictions = [int(terms_mtc_model.are_related(inst[TEXT_1_COL], inst[TEXT_2_COL])) for inst in tqdm(test_ds, desc="Prediction on Test Set")]
    test_results = compute_performance_clf(test_ds[LABEL_COL], predictions)
    return {
        "model_config": model_config,
        "test_results": {
            "test_performance": {k: None if np.isnan(v) else v for k, v in test_results.items()},
            "test_predictions": predictions
        }
    }, terms_mtc_model
