import os
from enum import StrEnum
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
from transformers import AutoModel, Cache, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from vuteco.core.common.constants import (UNSLOTH_TRAINING_KEY,
                                          E2EArchitectureStyle)
from vuteco.core.modeling.modeling_common import load_model_with_tokenizer
from vuteco.core.modeling.modeling_lm_lnk import (CodeLlamaLinker,
                                                  DeepSeekCoderLinker,
                                                  LanguageModelLinker,
                                                  LanguageModelLinkerConfig,
                                                  QwenCoderLinker)


class LanguageModelE2EConfigKeys(StrEnum):
    UNSLOTH_TRAINING = UNSLOTH_TRAINING_KEY
    ARCHI_STYLE = "archi_style"
    TRAIN_TYPE = "train_type"
    FT_AUGMENT_TECH = "ft_augment_technique"
    USE_CWE = "use_cwe"
    LNK_AUGMENT_TECH = "lnk_augment_technique"
    # Hyperparams
    LNK_AUGMENT_EXT = "lnk_augment_extent"
    LNK_EPOCHS = "lnk_epochs"
    FT_AUGMENT_EXT = "ft_augment_extent"
    FT_EPOCHS = "ft_epochs"


class LanguageModelE2EConfig(PretrainedConfig):
    model_type = "end2end"

    def __init__(self,
                 archi_style=E2EArchitectureStyle.LINKER_ONLY,
                 use_cwe=False,
                 unsloth_training=False,
                 **kwargs):
        self.archi_style = archi_style
        self.use_cwe = use_cwe
        self.unsloth_training = unsloth_training
        # NOTE LLM - Decide the LoRA configuration elements to experiment, if needed
        super().__init__(**kwargs)


class LanguageModelE2E(PreTrainedModel):
    # Must be there to make the model loading work correctly
    config_class = LanguageModelE2EConfig
    supports_gradient_checkpointing = True
    model_max_length = 4096
    task_text = "You are an expert in unit testing and security testing. Given the following vulnerability description and JUnit test method (it might be truncated if too long), answer with 1 if the test case is likely to identify the described vulnerability in the code under test, or 0 if it is not. Answer with only the number, with no explanation."

    def __init__(self,
                 config: LanguageModelE2EConfig,
                 lnk_model_class: Type[LanguageModelLinker],
                 lnk_model: LanguageModelLinker = None,
                 cache_dir: str = None):
        super().__init__(config)
        self.config = config
        if self.config.archi_style == E2EArchitectureStyle.LINKER_ONLY:
            if lnk_model:
                self.lm_model: LanguageModelLinker = lnk_model
            else:
                lnk_config_args = {
                    "use_cwe": config.use_cwe,
                    "unsloth_training": config.unsloth_training
                }
                self.lm_model = lnk_model_class(
                    LanguageModelLinkerConfig(**{k: v for k, v in lnk_config_args.items() if v is not None}),
                    cache_dir=cache_dir
                )
            self.tokenizer = self.lm_model.tokenizer
            self.response_starts_after = self.lm_model.response_starts_after
        else:
            raise NotImplementedError("Cannot build the model with this architecture style!")
        self.base_model_prefix = self.lm_model.base_model_prefix

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.config.archi_style == E2EArchitectureStyle.LINKER_ONLY:
            return self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs
            )
        else:
            raise NotImplementedError("Cannot run forward with this architecture style!")

    def get_input_embeddings(self):
        return self.lm_model.lm_model.get_input_embeddings()

    def prepare_batch_train(self, instance_batch, **kwargs):
        return self.lm_model.prepare_batch_train(instance_batch, **kwargs)

    def prepare_batch_eval(self, instance_batch, **kwargs):
        return self.lm_model.prepare_batch_eval(instance_batch, **kwargs)

    def do_generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        return self.lm_model.do_generate(input_ids, attention_mask)

    def extract_responses(self, token_ids: Union[np.ndarray, torch.Tensor]) -> Union[str, List[str]]:
        return self.lm_model.extract_responses(token_ids)

    def get_relation_score(self, test_code: Union[str, list[str]], vuln: str) -> Union[float, list[float]]:
        return self.lm_model.get_relation_score(test_code, vuln)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None and not os.path.exists(os.path.join(save_directory, "tokenizer.json")):
            self.tokenizer.save_pretrained(save_directory)
        if self.config is not None and not os.path.exists(os.path.join(save_directory, "config.json")):
            self.config.save_pretrained(save_directory)

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], model_class: Type[PreTrainedModel], *model_args, **kwargs) -> 'LanguageModelE2E':
    #     return load_model_with_tokenizer(pretrained_model_name_or_path, model_class.from_pretrained, *model_args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'LanguageModelE2E':
        lnk_dir = kwargs.get("lnk_dir", None)
        lnk_model = AutoModel.from_pretrained(lnk_dir) if lnk_dir else None
        kwargs["lnk_model"] = lnk_model
        return load_model_with_tokenizer(pretrained_model_name_or_path, super().from_pretrained, *model_args, **kwargs)


class CodeLlamaE2EModel(LanguageModelE2E):
    LNK_MODEL_CLASS = CodeLlamaLinker

    def __init__(self,
                 config: LanguageModelE2EConfig,
                 lnk_model: LanguageModelLinker = None,
                 cache_dir: str = None):
        super().__init__(config,
                         lnk_model_class=self.LNK_MODEL_CLASS,
                         lnk_model=lnk_model,
                         cache_dir=cache_dir)
        self.post_init()

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'CodeLlamaE2EModel':
    #     super().from_pretrained(pretrained_model_name_or_path, LlamaForCausalLM, *model_args, **kwargs)


class QwenCoderE2EModel(LanguageModelE2E):
    LNK_MODEL_CLASS = QwenCoderLinker

    def __init__(self,
                 config: LanguageModelE2EConfig,
                 lnk_model: LanguageModelLinker = None,
                 cache_dir: str = None):
        super().__init__(config,
                         lnk_model_class=self.LNK_MODEL_CLASS,
                         lnk_model=lnk_model,
                         cache_dir=cache_dir)
        self.post_init()

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'QwenCoderE2EModel':
    #     super().from_pretrained(pretrained_model_name_or_path, Qwen2ForCausalLM, *model_args, **kwargs)


class DeepSeekCoderE2EModel(LanguageModelE2E):
    LNK_MODEL_CLASS = DeepSeekCoderLinker

    def __init__(self,
                 config: LanguageModelE2EConfig,
                 lnk_model: LanguageModelLinker = None,
                 cache_dir: str = None):
        super().__init__(config,
                         lnk_model_class=self.LNK_MODEL_CLASS,
                         lnk_model=lnk_model,
                         cache_dir=cache_dir)
        self.post_init()

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'DeepSeekCoderE2EModel':
    #     super().from_pretrained(pretrained_model_name_or_path, LlamaForCausalLM, *model_args, **kwargs)
