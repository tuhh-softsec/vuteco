import re
from typing import List, Union

import numpy as np
import torch
from core.common.constants import TEXT_COL
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

"""
Prepare the data in the "conversational" format for "language modeling" (see https://huggingface.co/docs/trl/v0.16.0/en/dataset_formats) and then applies the tokenizer's chat template. Optionally, we can also tokenize.
"""


def prepare_for_lm(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                   task_text: str,
                   text_1: Union[str, List[str], List[List[str]]],
                   text_2: Union[str, List[str], List[List[str]]] = None,
                   expected_text: Union[str, List[str], List[List[str]]] = None,
                   must_tokenize: bool = True,
                   max_length: int = None) -> dict[str, list]:
    # Loop as apply_chat_template works for one conversation at a time
    iter_text_1 = text_1 if isinstance(text_1, List) else [text_1]
    if text_2 is not None:
        iter_text_2 = text_2 if isinstance(text_2, List) else [text_2]
    else:
        iter_text_2 = [None] * len(iter_text_1)
    if expected_text is not None:
        iter_expected_text = expected_text if isinstance(expected_text, List) else [expected_text]
    else:
        iter_expected_text = [None] * len(iter_text_1)

    all_convs = []
    # NOTE LLM - If many warnings are printed regarding the model length ("This instance will be ignored in loss calculation, increase the `max_length`"), we should change the ratio.
    max_tokens_for_t_1 = int(max_length * 0.9) if max_length is not None else None
    for t_1, t_2, et in zip(iter_text_1, iter_text_2, iter_expected_text):
        # NOTE LLM - Linker: We might also try with other ways to concatenate, like the Javadoc trick, but this would require adjusting the task text (later, to support as a CLI argument).
        if max_tokens_for_t_1 is not None:
            t_1_input_ids = tokenizer.encode(t_1)
            if len(t_1_input_ids) > max_tokens_for_t_1:
                t_1_ok = tokenizer.decode(t_1_input_ids[:max_tokens_for_t_1], skip_special_tokens=True)
            else:
                t_1_ok = t_1
        if t_2 is None:
            user_prompt = f"JUnit Test Method:\n{t_1_ok}"
        else:
            user_prompt = f"Vulnerability Description:\n{t_2}\n\nJUnit Test Method:\n{t_1_ok}"
        conv_t = [
            {"role": "system", "content": task_text},
            {"role": "user", "content": user_prompt}
        ]
        if et is not None:
            conv_t.append({"role": "assistant", "content": et})
        all_convs.append(conv_t)

    if must_tokenize:
        tokenized_convs = tokenizer.apply_chat_template(all_convs,
                                                        add_generation_prompt=et is None,
                                                        tokenize=True,
                                                        return_tensors="pt",
                                                        padding="max_length",
                                                        max_length=max_length,
                                                        truncation=True,
                                                        return_dict=True,
                                                        # **extra,
                                                        # return_attention_mask=True
                                                        )
        return tokenized_convs
    else:
        return {TEXT_COL: [tokenizer.apply_chat_template(c, add_generation_prompt=et is None, tokenize=False) for c in all_convs]}
    # return BatchEncoding({
    #     INPUT_IDS: torch.stack([t_ids[INPUT_IDS] for t_ids in all_convs]),
    #     ATTENTION_MASK: torch.stack([t_ids[ATTENTION_MASK] for t_ids in all_convs])
    # })


def extract_responses(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], response_starts_after: str, token_ids: Union[np.ndarray, torch.Tensor]) -> Union[str, List[str]]:
    if not hasattr(token_ids, 'ndim'):
        return None
    if token_ids.ndim == 1:
        decoded_output: str = tokenizer.decode(token_ids)
        return decoded_output.split(response_starts_after)[1] if response_starts_after in decoded_output else ""
    elif token_ids.ndim == 2:
        responses: list[str] = tokenizer.batch_decode(token_ids)
        return [
            r.split(response_starts_after)[1] if response_starts_after in r else ""
            for r in responses
        ]
    else:
        return None


def has_1_in_response(response: Union[str, list[str]]) -> Union[bool, list[bool]]:
    if isinstance(response, list):
        booleans = []
        for r in response:
            match = re.search(r'\d', r)
            if not match:
                return False
            booleans.append("1" == match.group())
        return booleans
    else:
        match = re.search(r'\d', response)
        if not match:
            return False
        return "1" == match.group()
