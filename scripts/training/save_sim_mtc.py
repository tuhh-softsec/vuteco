import json
import os
from argparse import Namespace
from typing import Any

from common.constants import CONFIG_JSON
from modeling.modeling_sim_mtc import (SimilarityMatcher,
                                       SimilarityMatcherConfigKeys)


def get_sim_mtc_config(args: Namespace):
    return {
        SimilarityMatcherConfigKeys.EXTRACTOR: args.sim_extractor,
        SimilarityMatcherConfigKeys.THRESHOLD: args.sim_threshold,
        SimilarityMatcherConfigKeys.NR_KEYWORDS: args.keywords_nr,
        SimilarityMatcherConfigKeys.DEDUP: args.keywords_dedup if args.keywords_dedup in ["leve", "jaro", "seqm"] else "seqm",
    }, {}


def save_sim_mtc(config: dict[SimilarityMatcherConfigKeys, Any], export_dir: str = None, log_outfile: str = None, cache_dir: str = None) -> SimilarityMatcher:
    # A fake training, just to make sure the object can be instantiated is enough
    sim_mtc_model = SimilarityMatcher(**config, cache_dir=cache_dir)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        with open(os.path.join(export_dir, CONFIG_JSON), "w") as fout:
            json.dump(sim_mtc_model.get_config(), fout, indent=2)
    return sim_mtc_model
