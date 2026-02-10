import json
import os
import re
from enum import StrEnum
from typing import Optional

from vuteco.core.common.constants import (CONFIG_JSON, EXACT_PATTERN_FILEPATH,
                                          SUBSTRING_EXT_PATTERN_FILEPATH)


class GrepFinderConfigKeys(StrEnum):
    MATCHES = "matches"
    EXTENDED = "extended"


class GrepFinder():
    _matches: int
    _extended: bool
    _exact_patterns: list[re.Pattern]
    _substring_patterns: list[re.Pattern]

    def __init__(self, matches: int = 1, extended: bool = False) -> None:
        super().__init__()
        self._matches = matches
        self._extended = extended
        self._exact_patterns = []
        with open(SUBSTRING_EXT_PATTERN_FILEPATH) as fin:
            self._substring_patterns = [re.compile(p, flags=re.IGNORECASE) for p in fin.read().splitlines()]
        if self._extended:
            with open(EXACT_PATTERN_FILEPATH) as fin:
                self._exact_patterns.extend([re.compile(f"(^|\W|_){p}($|\W|_)", flags=re.IGNORECASE) for p in fin.read().splitlines()])
            with open(SUBSTRING_EXT_PATTERN_FILEPATH) as fin:
                self._substring_patterns.extend([re.compile(p, flags=re.IGNORECASE) for p in fin.read().splitlines()])

    def get_witnessing_score(self, test_code: str) -> float:
        return 1.0 if self.is_security_test(test_code) else 0.0

    def is_security_test(self, test_code: str, return_pattern: bool = False) -> tuple[bool, Optional[list[re.Pattern]]]:
        matches_done = 0
        matched_patterns = []
        for p in self._exact_patterns + self._substring_patterns:
            if p.search(test_code):
                matches_done += 1
                matched_patterns.append(p)
                if matches_done >= self._matches:
                    if return_pattern:
                        return True, matched_patterns
                    return True
        if return_pattern:
            return False, matched_patterns
        else:
            return False

    def get_config(self):
        return {
            GrepFinderConfigKeys.MATCHES: self._matches,
            GrepFinderConfigKeys.EXTENDED: self._extended,
        }

    @classmethod
    def load_grep_fnd(cls, where_dir: str) -> 'GrepFinder':
        with open(os.path.join(where_dir, CONFIG_JSON)) as fin:
            config: dict = json.load(fin)
        return cls(**config)
