import json
import os
from argparse import Namespace
from typing import Any

from vuteco.core.common.constants import CONFIG_JSON
from vuteco.core.modeling.modeling_terms_mtc import (TermsMatcher,
                                                     TermsMatcherConfigKeys)


def get_terms_mtc_config(args: Namespace):
    return {
        TermsMatcherConfigKeys.SEPARATE: args.separate,
        TermsMatcherConfigKeys.THRESHOLD: args.terms_threshold
    }, {}


def save_terms_mtc(config: dict[TermsMatcherConfigKeys, Any], export_dir: str = None, log_outfile: str = None) -> TermsMatcher:
    # A fake training, just to make sure the object can be instantiated is enough
    terms_mtc_model = TermsMatcher(**config)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        with open(os.path.join(export_dir, CONFIG_JSON), "w") as fout:
            json.dump(terms_mtc_model.get_config(), fout, indent=2)
    return terms_mtc_model
