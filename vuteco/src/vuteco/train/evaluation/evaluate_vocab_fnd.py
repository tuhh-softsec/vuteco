from typing import Any

import datasets
import numpy as np
from tqdm import tqdm

from vuteco.core.common.constants import LABEL_COL, TEXT_COL
from vuteco.core.common.utils_training import compute_performance_clf
from vuteco.core.modeling.modeling_vocab_fnd import (
    VocabularyFinder, VocabularyFinderConfigKeys)
from vuteco.train.training.fit_vocab_fnd import fit_vocab_fnd


def eval_vocab_fnd(train_ds: datasets.Dataset,
                   test_ds: datasets.Dataset,
                   model_config: dict[VocabularyFinderConfigKeys, Any],
                   log_outfile: str,
                   hyperparam_space: dict[VocabularyFinderConfigKeys, list[Any]] = None) -> tuple[dict, VocabularyFinder]:
    vocab_fnd_model = fit_vocab_fnd(train_ds=train_ds, config=model_config, log_outfile=log_outfile)
    predictions = [int(vocab_fnd_model.is_security_test(inst[TEXT_COL])) for inst in tqdm(test_ds, desc="Prediction on Test Set")]
    test_results = compute_performance_clf(test_ds[LABEL_COL], predictions)
    return {
        "model_config": model_config,
        "test_results": {
            "test_performance": {k: None if np.isnan(v) else v for k, v in test_results.items()},
            "test_predictions": predictions,
            "test_matches": [vocab_fnd_model.get_matches(inst[TEXT_COL]) for inst in test_ds]
        }
    }, vocab_fnd_model
