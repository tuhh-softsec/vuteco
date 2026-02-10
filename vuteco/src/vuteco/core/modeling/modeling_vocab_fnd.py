import json
import os
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum

import yake

from vuteco.core.common.constants import CONFIG_JSON, NAMES_JSON
from vuteco.core.common.utils_mining import extract_identifiers


class VocabularyFinderConfigKeys(StrEnum):
    EXTRACTOR = "extractor"
    MATCHES = "matches"
    SEPARATE = "separate"
    DEDUP = "dedup"
    NR_KEYWORDS = "nr_kw"


class VocabularyFinder():
    extractor: str
    matches: int
    separate: bool
    kw_extractor: yake.KeywordExtractor
    names: dict[str, int]

    def __init__(self, extractor: str = "iden", matches: int = 1, separate: bool = False, dedup: str = "seqm", nr_kw: int = 20, names: dict[str, int] = {}) -> None:
        super().__init__()
        self.extractor = extractor
        self.matches = matches
        if self.extractor == "iden":
            self.separate = separate
        elif self.extractor == "yake":
            self.nr_keywords = nr_kw
            self.dedup = dedup
            self.kw_extractor = yake.KeywordExtractor(n=1, top=nr_kw, dedupFunc=dedup)
        self.names = names

    def fit(self, training_set: Iterable[str]):
        if self.extractor == "iden":
            terms = [id.lower() for inst in training_set for id in extract_identifiers(inst, self.separate)]
        elif self.extractor == "yake":
            terms = [kw[0].lower() for inst in training_set for kw in self.kw_extractor.extract_keywords(inst)]
        self.names = Counter(terms)

    def get_matches(self, test_code: str) -> int:
        if self.extractor == "iden":
            names = [id.lower() for id in extract_identifiers(test_code, self.separate)]
        elif self.extractor == "yake":
            names = [kw[0].lower() for kw in self.kw_extractor.extract_keywords(test_code)]
        names = list(dict.fromkeys(names))
        return sum(1 for name in names if self.is_in_vocab(name))

    def is_security_test(self, test_code: str) -> bool:
        return self.get_matches(test_code) >= self.matches

    def get_witnessing_score(self, test_code: str) -> float:
        return 1.0 if self.is_security_test(test_code) else 0.0

    def is_in_vocab(self, name: str):
        return name in self.names

    def get_occurrences(self, name: str):
        return self.names[name] if name in self.names else 0

    def get_config(self):
        return {
            VocabularyFinderConfigKeys.EXTRACTOR: self.extractor,
            VocabularyFinderConfigKeys.MATCHES: self.matches,
            VocabularyFinderConfigKeys.SEPARATE: self.separate,
            VocabularyFinderConfigKeys.NR_KEYWORDS: self.nr_keywords,
            VocabularyFinderConfigKeys.DEDUP: self.dedup,
        }

    def get_names(self):
        return self.names.copy()

    @classmethod
    def load_vocab_fnd(cls, where_dir: str) -> 'VocabularyFinder':
        with open(os.path.join(where_dir, NAMES_JSON)) as fin:
            names: dict = json.load(fin)
        with open(os.path.join(where_dir, CONFIG_JSON)) as fin:
            config: dict = json.load(fin)
        return cls(**config, names=names)
