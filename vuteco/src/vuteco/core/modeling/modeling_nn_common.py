from typing import List, Union

import torch
from transformers import (BatchEncoding, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from transformers.activations import gelu

from vuteco.core.common.constants import ATTENTION_MASK, INPUT_IDS

"""
Very similar to RobertaClassificationHead in https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py to have the freedom to modify the architecture. Inspired from https://github.com/awsm-research/LineVul/blob/30429b44f71b7c2c6d64a0180d0e59a795a4d5e3/linevul/linevul_model.py
"""


class ClassificationHead(torch.nn.Module):
    def __init__(self, in_size, hidden_size_1, hidden_size_2, out_size, dropout_p):
        super().__init__()
        self.activation = gelu
        self.dropout = torch.nn.Dropout(dropout_p)
        self.linear_layer_1 = torch.nn.Linear(in_size, hidden_size_1)
        if hidden_size_2 is not None and hidden_size_2 > 0:
            out_proj_in_size = hidden_size_2
            self.linear_layer_2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        else:
            out_proj_in_size = hidden_size_1
            self.linear_layer_2 = None
        self.out_proj = torch.nn.Linear(out_proj_in_size, out_size)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.linear_layer_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.linear_layer_2:
            x = self.linear_layer_2(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.out_proj(x)
        return x


def call_tokenizer_standard(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                            text_1: Union[str, List[str], List[List[str]]],
                            text_2: Union[str, List[str], List[List[str]]] = None,
                            truncate: bool = True) -> BatchEncoding:
    text_pair_arg = {"text_pair": text_2} if text_2 is not None else {}
    # When "attention_mask" (verbatim) in not in the tokenizer's model_input_names, the __call__ method won't compute the attention masks by default. So, to be safe, we explicitly ask to return attention masks in all cases.
    return tokenizer(text=text_1,
                     **text_pair_arg,
                     padding="max_length",
                     max_length=tokenizer.model_max_length,
                     truncation=truncate,
                     return_tensors="pt",
                     return_attention_mask=True)


def call_tokenizer_unixcoder(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                             text_1: Union[str, List[str], List[List[str]]],
                             text_2: Union[str, List[str], List[List[str]]] = None,
                             truncate: bool = True) -> BatchEncoding:
    if isinstance(text_1, str):
        pair_arg = {"pair": text_2} if text_2 is not None else {}
        # Readaptation from UniXCoder.tokenize()
        # Since UniXCoder forgot to add the model_max_length in the tokenizer configuration, we hardcode 512 (as shown in their usage examples)
        tokenizer_model_max_length = 512
        tokens = tokenizer.tokenize(text=text_1,
                                    **pair_arg,
                                    padding="max_length",
                                    max_length=tokenizer_model_max_length,
                                    truncation=truncate,
                                    return_tensors="pt")
        tokens = tokens[:tokenizer_model_max_length - 4]
        tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids + [tokenizer.pad_token_id] * (tokenizer_model_max_length - len(token_ids))
        # Manually create the BatchEncoding object, containing input_ids and attention_mask
        # be = tokenizer.prepare_for_model(ids=token_ids,
        #                                padding="max_length",
        #                                max_length=tokenizer_model_max_length,
        #                                truncation=truncate,
        #                                return_tensors="pt")
        # Fix attention mask, as its is not computed correctly
        # be[ATTENTION_MASK] = be[INPUT_IDS].ne(tokenizer.pad_token_id).long()
        token_ids_torch = torch.LongTensor(token_ids)
        be = BatchEncoding({
            INPUT_IDS: token_ids_torch,
            ATTENTION_MASK: token_ids_torch.ne(tokenizer.pad_token_id).long()
        })
        return be
    elif isinstance(text_1, List):
        # The list of BatchEncoding(s) must be converted inot a single BatchEncoding object
        bes = [call_tokenizer_unixcoder(tokenizer, i, truncate=truncate) for i in text_1]
        return BatchEncoding({
            INPUT_IDS: torch.stack([be[INPUT_IDS] for be in bes]),
            ATTENTION_MASK: torch.stack([be[ATTENTION_MASK] for be in bes])
        })

# Credits to https://gist.github.com/avidale/364ebc397a6d3425b1eba8cf2ceda525


def mean_pooling(x: torch.Tensor, attention_mask=None):
    if attention_mask is None:
        return x.mean(dim=1)
    token_embeddings = x
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def unixcoder_sentence_embeddings(embeddings, attention_mask=None):
    # Taken from UniXCoder.forward()
    return (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)


def cls_embeddings(embeddings, attention_mask=None):
    # With [:, 0, :] we extract embedding of [CLS] token
    return embeddings[:, 0, :]
