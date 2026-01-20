import json
import os
from argparse import Namespace
from typing import Any

from common.constants import CONFIG_JSON
from modeling.modeling_grep_fnd import GrepFinderConfigKeys, GrepFinder

def get_grep_fnd_config(args: Namespace):
    return {
        GrepFinderConfigKeys.MATCHES: args.grep_matches,
        GrepFinderConfigKeys.EXTENDED: args.grep_extended
    }, {}


def save_grep_fnd(config: dict[GrepFinderConfigKeys, Any], export_dir: str = None, log_outfile: str = None) -> GrepFinder:
    # A fake training, just to make sure the object can be instantiated is enough
    grep_fnd_model = GrepFinder(**config)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        with open(os.path.join(export_dir, CONFIG_JSON), "w") as fout:
            json.dump(grep_fnd_model.get_config(), fout, indent=2)
    return grep_fnd_model
