import json
import os
from argparse import Namespace
from typing import Any

from vuteco.core.common.constants import CONFIG_JSON
from vuteco.core.modeling.modeling_grep_mtc import (GrepMatcher,
                                                    GrepMatcherConfigKeys)


def get_grep_mtc_config(args: Namespace):
    return {
        GrepMatcherConfigKeys.MATCHES: args.grep_matches,
        GrepMatcherConfigKeys.EXTENDED: args.grep_extended
    }, {}


def save_grep_mtc(config: dict[GrepMatcherConfigKeys, Any], export_dir: str = None, log_outfile: str = None) -> GrepMatcher:
    # A fake training, just to make sure the object can be instantiated is enough
    grep_mtc_model = GrepMatcher(**config)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        with open(os.path.join(export_dir, CONFIG_JSON), "w") as fout:
            json.dump(grep_mtc_model.get_config(), fout, indent=2)
    return grep_mtc_model
