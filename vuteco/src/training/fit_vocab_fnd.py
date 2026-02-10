import json
import os
from argparse import Namespace
from typing import Any

from common.constants import CONFIG_JSON, LABEL_COL, NAMES_JSON, TEXT_COL
from datasets import Dataset
from modeling.modeling_vocab_fnd import (VocabularyFinderConfigKeys,
                                         VocabularyFinder)


def get_vocab_fnd_config(args: Namespace):
    return {
        VocabularyFinderConfigKeys.EXTRACTOR: args.vocab_extractor,
        VocabularyFinderConfigKeys.SEPARATE: args.separate,
        VocabularyFinderConfigKeys.MATCHES: args.vocab_matches,
        VocabularyFinderConfigKeys.NR_KEYWORDS: args.keywords_nr,
        VocabularyFinderConfigKeys.DEDUP: args.keywords_dedup if args.keywords_dedup in ["leve", "jaro", "seqm"] else "seqm",
    }, {}


def fit_vocab_fnd(train_ds: Dataset, config: dict[VocabularyFinderConfigKeys, Any], export_dir: str = None, log_outfile: str = None) -> VocabularyFinder:
    training_set = [t[TEXT_COL] for t in train_ds if t[LABEL_COL]]
    vocab_fnd_model = VocabularyFinder(**config)
    vocab_fnd_model.fit(training_set=training_set)
    #print_stdout_file("Fitted Vocabulary:", log_outfile)
    #print_stdout_file(dict(sorted(vocab_fnd_model.get_names().items(), key=lambda item: item[1], reverse=True)), log_outfile)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        with open(os.path.join(export_dir, NAMES_JSON), "w") as fout:
            json.dump(vocab_fnd_model.get_names(), fout, indent=2)
        with open(os.path.join(export_dir, CONFIG_JSON), "w") as fout:
            json.dump(vocab_fnd_model.get_config(), fout, indent=2)
    return vocab_fnd_model
