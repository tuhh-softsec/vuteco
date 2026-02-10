import json
import os
from enum import Enum
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch

from vuteco.core.common import global_vars
from vuteco.core.common.constants import (ATTENTION_MASK, DETERMINISM,
                                          INPUT_IDS, LABEL_COL, TEXT_1_COL,
                                          TEXT_2_COL, UNSLOTH_TRAINING_KEY)

if global_vars.MUST_LOAD_UNSLOTH:
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        UNSLOTH_LOADED = True
    except:
        UNSLOTH_LOADED = False
else:
    UNSLOTH_LOADED = False

from transformers import (AutoTokenizer, Cache, LlamaForCausalLM,
                          PretrainedConfig, PreTrainedModel, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.tokenization_llama import E_INST

from vuteco.core.modeling.modeling_common import load_model_with_tokenizer
from vuteco.core.modeling.modeling_lm_common import (extract_responses,
                                                     has_1_in_response,
                                                     prepare_for_lm)


class LanguageModelLinkerConfigKeys(str, Enum):
    UNSLOTH_TRAINING = UNSLOTH_TRAINING_KEY
    USE_CWE = "use_cwe"
    AUGMENT_TECH = "augment_technique"
    # LOSS = "loss"
    # Hyperparams
    AUGMENT_EXT = "augment_extent"
    EPOCHS = "epochs"


class LanguageModelLinkerConfig(PretrainedConfig):
    model_type = "linker"

    def __init__(self,
                 # loss_fun=LossFunction.BCE,
                 #  class_weights=[1.0, 1.0],
                 use_cwe=False,
                 unsloth_training=False,
                 **kwargs):
        # self.loss_fun = loss_fun
        #  self.class_weights = class_weights
        self.use_cwe = use_cwe
        self.unsloth_training = unsloth_training
        # NOTE LLM - Decide the LoRA configuration elements to experiment, if needed
        super().__init__(**kwargs)


class LanguageModelLinker(PreTrainedModel):
    # Must be there to make the model loading work correctly
    config_class = LanguageModelLinkerConfig
    supports_gradient_checkpointing = True
    model_max_length = 4096
    task_text = "You are an expert in unit testing and security testing. Given the following vulnerability description and JUnit test method (it might be truncated if too long), answer with 1 if the test case is likely to identify the described vulnerability in the code under test, or 0 if it is not. Answer with only the number, with no explanation."

    def __init__(self, config: LanguageModelLinkerConfig, model_name: str, model_class: Type[PreTrainedModel], unsloth_chat_template_name: str = None, cache_dir: str = None, trust_remote_code: bool = False, hf_model_download: bool = True):
        super().__init__(config)
        self.config = config
        if self.config.unsloth_training and UNSLOTH_LOADED:
            self.lm_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                # max_seq_length=2048, # Choose any! We auto support RoPE Scaling internally!
                dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
                load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False.
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
            # NOTE LLM - Unsloth prefers to set the chat template for the models it supports. If a model is not supported by Unsloth, we use the default templated define in the tokenizer.
            if unsloth_chat_template_name is not None:
                print("Tokenizer Default Chat Template:")
                print(self.tokenizer.chat_template)
                print("Padding side:")
                print(self.tokenizer.padding_side)
                self.tokenizer = get_chat_template(
                    self.tokenizer,
                    chat_template=unsloth_chat_template_name,
                )
                print("Unsloth Chat Template:")
                print(self.tokenizer.chat_template)
                print("Padding side:")
                print(self.tokenizer.padding_side)
            else:
                raise ValueError("No viable chat template was set for the Unsloth model: We cannot proceed!")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
            if hf_model_download:
                self.lm_model = model_class.from_pretrained(model_name, torch_dtype="auto", cache_dir=cache_dir, trust_remote_code=trust_remote_code, use_safetensors=True)
                self.base_model_prefix = self.lm_model.base_model_prefix
            else:
                self.lm_model = None
        self.response_starts_after = ""
        # if self.config.loss_fun == LossFunction.WBCE:
        #     self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.config.class_weights))
        # else:
        #     self.loss_fct = torch.nn.CrossEntropyLoss()

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

    def prepare_batch_train(self, instance_batch, **kwargs):
        expected = instance_batch[LABEL_COL] if LABEL_COL in instance_batch else None
        if expected is not None:
            expected = [str(e) for e in expected]
        # During training, SFTTrainer will take care of tokenizing (i.e., creating input_ids) for us. So, we don't tokenize for training.
        return prepare_for_lm(self.tokenizer,
                              task_text=self.task_text,
                              text_1=instance_batch[TEXT_1_COL],
                              text_2=instance_batch[TEXT_2_COL],
                              expected_text=expected,
                              must_tokenize=False,
                              max_length=self.model_max_length)

    def prepare_batch_eval(self, instance_batch, **kwargs):
        return prepare_for_lm(self.tokenizer,
                              task_text=self.task_text,
                              text_1=instance_batch[TEXT_1_COL],
                              text_2=instance_batch[TEXT_2_COL],
                              max_length=self.model_max_length)

    def do_generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Simple configuration, we're not interested in the case without determinism for now
        if not DETERMINISM:
            extra = {
                "temperature": 0.1,
                "top_p": 0.9,
                "min_p": 0.1
            }
        else:
            extra = {
                "temperature": None,
                "top_p": None,
                "top_k": None,
            }
        return self.lm_model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      max_new_tokens=2,
                                      use_cache=True,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                      do_sample=not DETERMINISM,
                                      **extra,
                                      )

    def extract_responses(self, token_ids: Union[np.ndarray, torch.Tensor]) -> Union[str, List[str]]:
        return extract_responses(self.tokenizer, self.response_starts_after, token_ids)

    def get_relation_score(self, test_code: Union[str, list[str]], vuln: str) -> Union[float, list[float]]:
        batched = isinstance(test_code, list)
        vulns = [vuln] * len(test_code) if batched else vuln
        model_input = prepare_for_lm(self.tokenizer,
                                     task_text=self.task_text,
                                     text_1=test_code,
                                     text_2=vulns,
                                     max_length=self.model_max_length)
        self.eval()
        torch.cuda.empty_cache()
        with torch.inference_mode():
            input_ids = model_input[INPUT_IDS].to(self.lm_model.device)
            attention_mask = model_input[ATTENTION_MASK].to(self.lm_model.device)
            output_ids = self.do_generate(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
        responses = self.extract_responses(output_ids)
        del input_ids, attention_mask, output_ids
        torch.cuda.empty_cache()
        # NOTE LLM - An LLM is not instructed to predict probabilities on its own. Unless we call it on the same prompt for X times and without DETERMINISM, so that we can count the nr. times a certain label has been predicted. For now, we set the probability to 1.0, always
        if batched:
            return [1.0 * int(c) for c in has_1_in_response(responses)]
        else:
            if isinstance(responses, list):
                responses = responses[0]
            return 1.0 * int(has_1_in_response(responses))

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None and not os.path.exists(os.path.join(save_directory, "tokenizer.json")):
            self.tokenizer.save_pretrained(save_directory)
        if self.config is not None and not os.path.exists(os.path.join(save_directory, "config.json")):
            self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'LanguageModelLinker':
        # with open(os.path.join(pretrained_model_name_or_path, "config.json")) as fin:
        #     config = LanguageModelLinkerConfig(json.load(fin))
        # model_name = model_class.UNSLOTH_MODEL_NAME if config.unsloth_training else model_class.PLAIN_MODEL_NAME
        # kwargs["trust_remote_code"] = True
        # kwargs["torch_dtype"] = "auto"
        # base_model = load_model_and_tokenizer(model_name, model_class.from_pretrained, *model_args, **kwargs)
        # return PeftModel.from_pretrained(base_model, pretrained_model_name_or_path)
        return load_model_with_tokenizer(pretrained_model_name_or_path, super().from_pretrained, *model_args, **kwargs)


class CodeLlamaLinker(LanguageModelLinker):
    # TODO LLM - "unsloth/codellama-7b-bnb-4bit" is based on the non-instruction-tuned codellama. When we will create the non-instruction-tuned variant, change, so that we don't mix models.
    UNSLOTH_MODEL_NAME = "unsloth/codellama-7b-bnb-4bit"
    PLAIN_MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"

    def __init__(self, config: LanguageModelLinkerConfig, cache_dir: str = None, hf_model_download: bool = True):
        model_name = self.UNSLOTH_MODEL_NAME if config.unsloth_training else self.PLAIN_MODEL_NAME
        super().__init__(config, model_name=model_name, model_class=LlamaForCausalLM, unsloth_chat_template_name="llama", cache_dir=cache_dir, hf_model_download=hf_model_download)
        self.response_starts_after = E_INST
        # This is a hotfix for Codellama as it didn't define a specific token for padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # SFTTrainer expects padding_side to right, but we must definitely have this to left for correct inference.
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        self.post_init()

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'CodeLlamaLinker':
    #     super().from_pretrained(pretrained_model_name_or_path, LlamaForCausalLM, *model_args, **kwargs)


class QwenCoderLinker(LanguageModelLinker):
    UNSLOTH_MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
    PLAIN_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

    def __init__(self, config: LanguageModelLinkerConfig, cache_dir: str = None, hf_model_download: bool = True):
        model_name = self.UNSLOTH_MODEL_NAME if config.unsloth_training else self.PLAIN_MODEL_NAME
        super().__init__(config, model_name=model_name, model_class=Qwen2ForCausalLM, unsloth_chat_template_name="qwen-2.5", cache_dir=cache_dir, hf_model_download=hf_model_download)
        self.response_starts_after = "<|im_start|>assistant\n"
        # SFTTrainer expects padding_side to right, but we must definitely have this to left for correct inference.
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        self.post_init()

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'QwenCoderLinker':
    #     super().from_pretrained(pretrained_model_name_or_path, Qwen2ForCausalLM, *model_args, **kwargs)


class DeepSeekCoderLinker(LanguageModelLinker):
    UNSLOTH_MODEL_NAME = None
    PLAIN_MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"

    def __init__(self, config: LanguageModelLinkerConfig, cache_dir: str = None, hf_model_download: bool = True):
        model_name = self.UNSLOTH_MODEL_NAME if config.unsloth_training else self.PLAIN_MODEL_NAME
        super().__init__(config, model_name=model_name, model_class=LlamaForCausalLM, cache_dir=cache_dir, trust_remote_code=True, hf_model_download=hf_model_download)
        self.response_starts_after = "### Response:\n"
        # SFTTrainer expects padding_side to right, but we must definitely have this to left for correct inference.
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        self.post_init()

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'DeepSeekCoderLinker':
    #     super().from_pretrained(pretrained_model_name_or_path, LlamaForCausalLM, *model_args, **kwargs)
