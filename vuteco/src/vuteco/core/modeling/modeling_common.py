import os
from collections.abc import Callable
from enum import StrEnum
from typing import Optional, Union

from transformers import AutoTokenizer

from vuteco.core.common.constants import DEVICE


class FinderLabel(StrEnum):
    UNKNOWN = "Unknown"
    WITNESSING = "Witnessing"


class LinkerLabel(StrEnum):
    NOT_LINKED = "NotLinked"
    LINKED = "Linked"


class E2ELabel(StrEnum):
    NOT_RELATED = "NotRelated"
    RELATED = "Related"


def load_model_with_tokenizer(pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], model_load_fn: Callable, *model_args, **kwargs):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = model_load_fn(pretrained_model_name_or_path, *model_args, **kwargs)
    #if model.lm_model is not None and model.base_model_prefix is None:
    #    model.base_model_prefix = model.lm_model.base_model_prefix
    model.to(DEVICE)
    if not hasattr(model, "tokenizer") or model.tokenizer is None:
        model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, num_labels=model.config.num_labels)
    return model