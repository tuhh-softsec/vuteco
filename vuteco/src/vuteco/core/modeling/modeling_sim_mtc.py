import json
import os
from enum import StrEnum
from typing import Union

import torch
import yake
from transformers import (AutoModel, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, PreTrainedTokenizerFast)

from vuteco.core.common.constants import (ATTENTION_MASK, CODEBERT_FULL,
                                          CODET5PLUS_FULL, CONFIG_JSON,
                                          INPUT_IDS, UNIXCODER_FULL)
from vuteco.core.modeling.modeling_nn_common import (
    call_tokenizer_standard, call_tokenizer_unixcoder, cls_embeddings,
    mean_pooling, unixcoder_sentence_embeddings)


class SimilarityMatcherConfigKeys(StrEnum):
    EXTRACTOR = "extractor"
    THRESHOLD = "threshold"
    DEDUP = "dedup"
    NR_KEYWORDS = "nr_kw"


class SimilarityMatcher():
    extractor: str
    threshold: float
    dedup: str
    nr_keywords: int
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    embedding_model: PreTrainedModel
    kw_extractor: yake.KeywordExtractor

    def __init__(self, extractor: str = CODEBERT_FULL, threshold: float = 0.5, dedup: str = "seqm", nr_kw: int = 20, cache_dir: str = None) -> None:
        super().__init__()
        self.extractor = extractor
        self.threshold = threshold
        self.nr_keywords = nr_kw
        self.dedup = dedup
        if self.extractor == CODEBERT_FULL:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", cache_dir=cache_dir)
            self.embedding_model = AutoModel.from_pretrained("microsoft/codebert-base", cache_dir=cache_dir)
        elif self.extractor == UNIXCODER_FULL:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base", cache_dir=cache_dir)
            self.embedding_model = AutoModel.from_pretrained("microsoft/unixcoder-base", cache_dir=cache_dir)
        elif self.extractor == CODET5PLUS_FULL:
            self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m", cache_dir=cache_dir)
            self.embedding_model = AutoModel.from_pretrained("Salesforce/codet5p-220m", cache_dir=cache_dir)
        elif self.extractor == "yake":
            self.kw_extractor = yake.KeywordExtractor(n=1, top=nr_kw, dedupFunc=dedup)

    def get_relation_score(self, test_code: str, vuln_descr: str) -> float:
        if self.extractor == CODEBERT_FULL:
            test_code_tokens = call_tokenizer_standard(self.tokenizer, test_code, truncate=True)
            vuln_descr_tokens = call_tokenizer_standard(self.tokenizer, vuln_descr, truncate=True)
            # test_code_tokens = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + self.tokenizer.tokenize(test_code) + [self.tokenizer.eos_token])
            # test_code_embedding = self.embedding_model(torch.tensor(test_code_tokens)[None, :])[0][:, 0, :]
            with torch.no_grad():
                test_code_embedding = cls_embeddings(self.embedding_model(**test_code_tokens).last_hidden_state)
                vuln_descr_embedding = cls_embeddings(self.embedding_model(**vuln_descr_tokens).last_hidden_state)
            # vuln_descr_tokens = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + self.tokenizer.tokenize(vuln_descr) + [self.tokenizer.eos_token])
            # vuln_descr_embedding = self.embedding_model(torch.tensor(vuln_descr_tokens)[None, :])[0][:, 0, :]
            sim_score = abs(torch.cosine_similarity(test_code_embedding, vuln_descr_embedding).item())
        elif self.extractor == UNIXCODER_FULL:
            test_code_tokens = call_tokenizer_unixcoder(self.tokenizer, test_code, truncate=True)
            test_code_tokens[INPUT_IDS] = test_code_tokens[INPUT_IDS].unsqueeze(0)
            test_code_tokens[ATTENTION_MASK] = test_code_tokens[ATTENTION_MASK].unsqueeze(0)
            vuln_descr_tokens = call_tokenizer_unixcoder(self.tokenizer, vuln_descr, truncate=True)
            vuln_descr_tokens[INPUT_IDS] = vuln_descr_tokens[INPUT_IDS].unsqueeze(0)
            vuln_descr_tokens[ATTENTION_MASK] = vuln_descr_tokens[ATTENTION_MASK].unsqueeze(0)
            with torch.no_grad():
                test_code_lhs = self.embedding_model(**test_code_tokens).last_hidden_state
                test_code_embedding = unixcoder_sentence_embeddings(test_code_lhs, test_code_tokens[ATTENTION_MASK])
                vuln_descr_lhs = self.embedding_model(**vuln_descr_tokens).last_hidden_state
                vuln_descr_embedding = unixcoder_sentence_embeddings(vuln_descr_lhs, vuln_descr_tokens[ATTENTION_MASK])
            sim_score = abs(torch.cosine_similarity(test_code_embedding, vuln_descr_embedding).item())
        elif self.extractor == CODET5PLUS_FULL:
            test_code_tokens = call_tokenizer_standard(self.tokenizer, test_code, truncate=True)
            vuln_descr_tokens = call_tokenizer_standard(self.tokenizer, vuln_descr, truncate=True)
            with torch.no_grad():
                test_code_lhs = self.embedding_model.encoder(**test_code_tokens, output_hidden_states=True, return_dict=True).hidden_states[-1]
                test_code_embedding = mean_pooling(test_code_lhs, test_code_tokens[ATTENTION_MASK])
                vuln_descr_lhs = self.embedding_model.encoder(**vuln_descr_tokens, output_hidden_states=True, return_dict=True).hidden_states[-1]
                vuln_descr_embedding = mean_pooling(vuln_descr_lhs, vuln_descr_tokens[ATTENTION_MASK])
            sim_score = abs(torch.cosine_similarity(test_code_embedding, vuln_descr_embedding).item())
        elif self.extractor == "yake":
            test_kw = set(kw[0] for kw in self.kw_extractor.extract_keywords(test_code))
            descr_kw = set(kw[0] for kw in self.kw_extractor.extract_keywords(vuln_descr))
            common_kw = test_kw.intersection(descr_kw)
            all_kw = test_kw.union(descr_kw)
            sim_score = float(len(common_kw)) / len(all_kw) if len(all_kw) > 0 else 0.0
        return sim_score

    def are_related(self, test_code: str, vuln_descr: str) -> bool:
        return self.get_relation_score(test_code, vuln_descr) >= self.threshold

    def is_score_beyond_threshold(self, score: float) -> bool:
        return score >= self.threshold

    def get_config(self):
        return {
            SimilarityMatcherConfigKeys.EXTRACTOR: self.extractor,
            SimilarityMatcherConfigKeys.THRESHOLD: self.threshold,
            SimilarityMatcherConfigKeys.NR_KEYWORDS: self.nr_keywords,
            SimilarityMatcherConfigKeys.DEDUP: self.dedup,
        }

    @classmethod
    def load_sim_mtc(cls, where_dir: str) -> 'SimilarityMatcher':
        with open(os.path.join(where_dir, CONFIG_JSON)) as fin:
            config: dict = json.load(fin)
        return cls(**config)
