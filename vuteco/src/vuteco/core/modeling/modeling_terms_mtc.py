import json
import os
from enum import Enum

from vuteco.core.common.constants import CONFIG_JSON
from vuteco.core.common.utils_mining import extract_identifiers


class TermsMatcherConfigKeys(str, Enum):
    SEPARATE = "separate"
    THRESHOLD = "threshold"


class TermsMatcher():
    _separate: bool
    _threshold: float

    def __init__(self, separate: bool = False, threshold: float = 0.5) -> None:
        super().__init__()
        self._separate = separate
        self._threshold = threshold

    def get_relation_score(self, test_code: str, vuln: str) -> float:
        return 1.0 if self.are_related(test_code, vuln) else 0.0

    def are_related(self, test_code: str, vuln_descr: str) -> bool:
        matches_done = 0
        dev_names = extract_identifiers(test_code, self._separate)
        for name in dev_names:
            if name.lower() in vuln_descr:
                matches_done += 1
        ratio = matches_done / len(dev_names) if len(dev_names) > 0 else 0.0
        return ratio >= self._threshold

    def get_config(self):
        return {
            TermsMatcherConfigKeys.SEPARATE: self._separate,
            TermsMatcherConfigKeys.THRESHOLD: self._threshold
        }

    @classmethod
    def load_terms_mtc(cls, where_dir: str) -> 'TermsMatcher':
        with open(os.path.join(where_dir, CONFIG_JSON)) as fin:
            config: dict = json.load(fin)
        return cls(**config)
