import os
from enum import StrEnum
from typing import List, Optional, Type, Union

import torch
from transformers import (AutoTokenizer, PretrainedConfig, PreTrainedModel,
                          RobertaModel, T5EncoderModel)
from transformers.modeling_outputs import SequenceClassifierOutput

from vuteco.core.common.constants import DEVICE, TEXT_COL, LossFunction
from vuteco.core.common.utils_training import one_line_text
from vuteco.core.modeling.modeling_common import (FinderLabel,
                                                  load_model_with_tokenizer)
from vuteco.core.modeling.modeling_nn_common import (
    ClassificationHead, call_tokenizer_standard, call_tokenizer_unixcoder,
    cls_embeddings, mean_pooling, unixcoder_sentence_embeddings)


class NeuralNetworkFinderConfigKeys(StrEnum):
    AUGMENT_TECH = "augment_technique"
    LOSS = "loss"
    ONE_LINE = "one_line"
    # Hyperparams
    AUGMENT_EXT = "augment_extent"
    HIDDEN_SIZE_1 = "hidden_size_1"
    HIDDEN_SIZE_2 = "hidden_size_2"
    EPOCHS = "epochs"


"""
Very similar to RobertaConfig in https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/configuration_roberta.py to have the freedom to modify the architecture. Suggested by the tutorial https://huggingface.co/docs/transformers/custom_models.
"""


class NeuralNetworkFinderConfig(PretrainedConfig):
    model_type = "finder"

    def __init__(self,
                 clf_hidden_size_1=768,
                 clf_hidden_size_2=512,
                 clf_dropout=0.1,
                 loss_fun=LossFunction.BCE,
                 class_weights=[1.0, 1.0],
                 one_line_text=False,
                 **kwargs):
        self.clf_hidden_size_1 = clf_hidden_size_1
        self.clf_hidden_size_2 = clf_hidden_size_2
        self.clf_dropout = clf_dropout
        self.loss_fun = loss_fun
        self.class_weights = class_weights
        self.one_line_text = one_line_text
        id2label = kwargs.pop("id2label", {0: FinderLabel.UNKNOWN.value, 1: FinderLabel.WITNESSING.value})
        label2id = kwargs.pop("label2id", {FinderLabel.UNKNOWN.value: 0, FinderLabel.WITNESSING.value: 1})
        super().__init__(num_labels=2, id2label=id2label, label2id=label2id, **kwargs)


"""
Similar to RobertaForSequenceClassification in https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py to have the freedom to modify the architecture. Inspired by https://github.com/awsm-research/LineVul/blob/30429b44f71b7c2c6d64a0180d0e59a795a4d5e3/linevul/linevul_model.py and in line with the tutorial https://huggingface.co/docs/transformers/custom_models
"""


class NeuralNetworkFinder(PreTrainedModel):
    # Must be there to make the model loading work correctly
    config_class = NeuralNetworkFinderConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: NeuralNetworkFinderConfig, model_name: str, model_class: Type[PreTrainedModel], cache_dir: str = None):
        super().__init__(config)
        self.config = config
        self.num_labels = self.config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.embedding_model = model_class.from_pretrained(model_name, cache_dir=cache_dir, num_labels=config.num_labels)
        self.base_model_prefix = self.embedding_model.base_model_prefix
        self._no_split_modules = self.embedding_model._no_split_modules
        self._keys_to_ignore_on_load_unexpected = self.embedding_model._keys_to_ignore_on_load_unexpected
        self._tied_weights_keys = self.embedding_model._tied_weights_keys

        self.clf_head = ClassificationHead(
            in_size=self.embedding_model.config.hidden_size,
            hidden_size_1=self.config.clf_hidden_size_1,
            hidden_size_2=self.config.clf_hidden_size_2,
            out_size=self.config.num_labels,
            dropout_p=self.config.clf_dropout
        )
        if self.config.loss_fun == LossFunction.WBCE:
            self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.config.class_weights))
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()
        # Why is this here? Needed?
        self.inference_pipeline = None

    def _extract_sentence_embedding(self, embeddings, attention_mask=None):
        pass

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
                ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        args = {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        outputs = self.embedding_model(
            input_ids,
            **{k: v for k, v in args.items() if v is not None}
        )
        # The first element in the output tensor is usually the last_hidden_state: We ignore the other info in outputs.
        x = self._extract_sentence_embedding(outputs[0], attention_mask=attention_mask)
        logits = self.clf_head(x)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def must_one_line_text(self) -> bool:
        return self.config.one_line_text

    def _call_tokenizer(self,
                        text_1: Union[str, List[str], List[List[str]]],
                        truncate: bool = True):
        return call_tokenizer_standard(self.tokenizer, text_1, truncate=truncate)

    def encode_batch(self, instance_batch, truncate: bool = True, **kwargs):
        if self.must_one_line_text():
            instance_batch[TEXT_COL] = [one_line_text(inst) for inst in instance_batch[TEXT_COL]]
        return self._call_tokenizer(instance_batch[TEXT_COL], truncate=truncate)

    def encode_single(self, test_code: str, truncate: bool = True, **kwargs):
        ready_test_code = one_line_text(test_code) if self.must_one_line_text() else test_code
        return self._call_tokenizer(ready_test_code, truncate=truncate)

    def get_witnessing_score(self, test_code: str) -> float:
        model_input = self.encode_single(test_code)
        self.eval()
        with torch.inference_mode():
            model_input = {n: model_input[n].to(DEVICE) if n in model_input else None for n in self.tokenizer.model_input_names}
            # NOTE: Hotfix for UnixCoder, as its encode_single returns tensors of [512], instead of [1,512] 
            model_input = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in model_input.items()}
            model_output = self(**model_input)
            logits: torch.Tensor = model_output.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        related_prob = float(probs[0, self.config.label2id[FinderLabel.WITNESSING]].item())
        return related_prob

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None and not os.path.exists(os.path.join(save_directory, "tokenizer.json")):
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'NeuralNetworkFinder':
        return load_model_with_tokenizer(pretrained_model_name_or_path, super().from_pretrained, *model_args, **kwargs)


class CodeBertFinder(NeuralNetworkFinder):
    def __init__(self, config: NeuralNetworkFinderConfig, cache_dir: str = None):
        super().__init__(config, model_name="microsoft/codebert-base", model_class=RobertaModel, cache_dir=cache_dir)
        self.post_init()

    def _init_weights(self, module):
        self.embedding_model._init_weights(module)
        super()._init_weights(module)

    def _extract_sentence_embedding(self, embeddings, attention_mask=None):
        return cls_embeddings(embeddings, attention_mask)


class CodeT5PlusFinder(NeuralNetworkFinder):
    def __init__(self, config: NeuralNetworkFinderConfig, cache_dir: str = None):
        super().__init__(config, model_name="Salesforce/codet5p-220m", model_class=T5EncoderModel, cache_dir=cache_dir)
        self.post_init()

    def _init_weights(self, module):
        self.embedding_model._init_weights(module)
        super()._init_weights(module)

    def _extract_sentence_embedding(self, embeddings, attention_mask=None):
        # TODO For better performance, we could consider adding a proper MeanPooling layer to learn how to aggregate (see https://gist.github.com/avidale/364ebc397a6d3425b1eba8cf2ceda525)
        return mean_pooling(embeddings, attention_mask)
        """
        First token embedding (not good for T5)
        x = embeddings[:, 0, :]
        """

        """
        # From T5ForSequenceClassification
        eos_mask = input_ids.eq(self.embedding_model.config.eos_token_id).to(embeddings.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = embeddings.shape
        x = embeddings[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        """


class UniXCoderFinder(NeuralNetworkFinder):
    def __init__(self, config: NeuralNetworkFinderConfig, cache_dir: str = None):
        super().__init__(config, model_name="microsoft/unixcoder-base", model_class=RobertaModel, cache_dir=cache_dir)
        # These post-construction actions are reimplemented taken from https://github.com/microsoft/CodeBERT/blob/master/UniXcoder/unixcoder.py, as UniXCoder is only a wrapper class, not a PreTrainedModel. We don't know if these are needed, but we them anyway
        self.register_buffer("bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024))
        self.tokenizer.add_tokens(["<mask0>"], special_tokens=True)
        self.post_init()

    def _init_weights(self, module):
        self.embedding_model._init_weights(module)
        super()._init_weights(module)

    def _call_tokenizer(self,
                        text_1: Union[str, List[str], List[List[str]]],
                        truncate: bool = True):
        # Override as UniXCoder does something slightly different, we use this custom helper function
        return call_tokenizer_unixcoder(self.tokenizer, text_1, truncate=truncate)

    def _extract_sentence_embedding(self, embeddings, attention_mask=None):
        return unixcoder_sentence_embeddings(embeddings, attention_mask)
