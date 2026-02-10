import json
import os
import re
import string
from enum import StrEnum
from typing import Optional

import nltk
from nltk.corpus import stopwords

from vuteco.core.common.constants import CONFIG_JSON


class GrepMatcherConfigKeys(StrEnum):
    MATCHES = "matches"
    EXTENDED = "extended"


class GrepMatcher():
    _matches: int
    _extended: bool

    def __init__(self, matches: int = 1, extended: bool = False) -> None:
        super().__init__()
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        self._matches = matches
        self._extended = extended

    def get_relation_score(self, test_code: str, vuln: str) -> float:
        return 1.0 if self.are_related(test_code, vuln) else 0.0

    def are_related(self, test_code: str, vuln_desc: str, return_pattern: bool = False) -> tuple[bool, Optional[list[re.Pattern]]]:
        tokens = nltk.word_tokenize(vuln_desc)
        stop_words = set(stopwords.words('english'))
        punctuations = set(string.punctuation)
        tokens = [word for word in tokens if word.lower() not in stop_words and word not in punctuations]
        patterns: list[re.Pattern] = []
        for tok in tokens:
            if "cve-" in tok.lower() and "-" in tok:
                parts = tok.split("-")
                parts.append(tok)
                parts.append(tok.replace("-", ""))
                parts.append(tok.replace("-", "_"))
                for p in parts:
                    patterns.append(re.compile(f"(^|\W|_){p}($|\W|_)", flags=re.IGNORECASE))
            elif self._extended:
                patterns.append(re.compile(f"(^|\W|_){tok}($|\W|_)", flags=re.IGNORECASE))
        matches_done = 0
        matched_patterns = []
        for p in patterns:
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
            GrepMatcherConfigKeys.MATCHES: self._matches,
            GrepMatcherConfigKeys.EXTENDED: self._extended,
        }

    @classmethod
    def load_grep_mtc(cls, where_dir: str) -> 'GrepMatcher':
        with open(os.path.join(where_dir, CONFIG_JSON)) as fin:
            config: dict = json.load(fin)
        return cls(**config)
