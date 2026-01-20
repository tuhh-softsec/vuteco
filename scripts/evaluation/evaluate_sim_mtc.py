from typing import Any

import datasets
import numpy as np
from common.constants import LABEL_COL, TEXT_1_COL, TEXT_2_COL
from common.utils_training import compute_performance_clf
from modeling.modeling_sim_mtc import (SimilarityMatcher,
                                       SimilarityMatcherConfigKeys)
from tqdm import tqdm


def eval_sim_mtc(train_ds: datasets.Dataset,
                 test_ds: datasets.Dataset,
                 model_config: dict[SimilarityMatcherConfigKeys, Any],
                 log_outfile: str,
                 hyperparam_space: dict[SimilarityMatcherConfigKeys, list[Any]] = None,
                 cache_dir: str = None) -> tuple[dict, SimilarityMatcher]:
    sim_mtc_model = SimilarityMatcher(**model_config, cache_dir=cache_dir)
    scores = [sim_mtc_model.get_relation_score(inst[TEXT_1_COL], inst[TEXT_2_COL]) for inst in tqdm(test_ds, desc="Prediction on Test Set")]
    predictions = [int(sim_mtc_model.is_score_beyond_threshold(s)) for s in scores]
    test_results = compute_performance_clf(test_ds[LABEL_COL], predictions)
    return {
        "model_config": model_config,
        "test_results": {
            "test_performance": {k: None if np.isnan(v) else v for k, v in test_results.items()},
            "test_predictions": predictions,
            "test_scores": scores
        }
    }, sim_mtc_model
