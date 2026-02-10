import os
from dataclasses import dataclass
from enum import StrEnum
from statistics import mean
from typing import List, Optional, Tuple, Type, Union

import torch
from transformers import (AutoModel, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)
from transformers.modeling_outputs import SequenceClassifierOutput

from vuteco.core.common.constants import (ATTENTION_MASK, DEVICE,
                                          FND_ATTENTION_MASK, FND_INPUT_IDS,
                                          INPUT_IDS, JAVADOC_MULTILINE,
                                          JAVADOC_ONE_LINE, LNK_ATTENTION_MASK,
                                          LNK_ATTENTION_MASK_1,
                                          LNK_ATTENTION_MASK_2, LNK_INPUT_IDS,
                                          LNK_INPUT_IDS_1, LNK_INPUT_IDS_2,
                                          TEXT_1_COL, TEXT_2_1_COL, TEXT_2_COL,
                                          E2EArchitectureStyle, LossFunction)
from vuteco.core.common.utils_mining import prepend_text
from vuteco.core.common.utils_training import one_line_text
from vuteco.core.modeling.modeling_common import (E2ELabel,
                                                  load_model_with_tokenizer)
from vuteco.core.modeling.modeling_nn_common import (ClassificationHead,
                                                     call_tokenizer_standard,
                                                     call_tokenizer_unixcoder)
from vuteco.core.modeling.modeling_nn_fnd import (CodeBertFinder,
                                                  CodeT5PlusFinder,
                                                  FinderLabel,
                                                  NeuralNetworkFinder,
                                                  NeuralNetworkFinderConfig,
                                                  UniXCoderFinder)
from vuteco.core.modeling.modeling_nn_lnk import (CodeBertLinker,
                                                  CodeT5PlusLinker,
                                                  NeuralNetworkLinker,
                                                  NeuralNetworkLinkerConfig,
                                                  UniXCoderLinker)


class NeuralNetworkE2EConfigKeys(StrEnum):
    ARCHI_STYLE = "archi_style"
    SUSPECT_THRESHOLD = "suspect_threshold"
    TRAIN_TYPE = "train_type"
    FT_AUGMENT_TECH = "ft_augment_technique"
    MERGE = "merge"
    USE_CWE = "use_cwe"
    FND_AUGMENT_TECH = "fnd_augment_technique"
    LNK_AUGMENT_TECH = "lnk_augment_technique"
    LOSS = "loss"
    # Hyperparams
    FND_AUGMENT_EXT = "fnd_augment_extent"
    FND_HIDDEN_SIZE_1 = "fnd_hidden_size_1"
    FND_HIDDEN_SIZE_2 = "fnd_hidden_size_2"
    FND_EPOCHS = "fnd_epochs"
    FND_LOSS = "fnd_loss"
    FND_ONE_LINE = "fnd_one_line"
    LNK_AUGMENT_EXT = "lnk_augment_extent"
    LNK_HIDDEN_SIZE_1 = "lnk_hidden_size_1"
    LNK_HIDDEN_SIZE_2 = "lnk_hidden_size_2"
    LNK_EPOCHS = "lnk_epochs"
    LNK_LOSS = "lnk_loss"
    LNK_ONE_LINE = "lnk_one_line"
    FT_AUGMENT_EXT = "ft_augment_extent"
    FT_EPOCHS = "ft_epochs"


@dataclass
class SequenceClassifierOutputE2E(SequenceClassifierOutput):
    fnd_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    fnd_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    lnk_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    lnk_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    lnk_hidden_states_1: Optional[Tuple[torch.FloatTensor, ...]] = None
    lnk_attentions_1: Optional[Tuple[torch.FloatTensor, ...]] = None
    lnk_hidden_states_2: Optional[Tuple[torch.FloatTensor, ...]] = None
    lnk_attentions_2: Optional[Tuple[torch.FloatTensor, ...]] = None


"""
Very similar to RobertaConfig in https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/configuration_roberta.py to have the freedom to modify the architecture. Suggested by the tutorial https://huggingface.co/docs/transformers/custom_models.
"""


class NeuralNetworkE2EConfig(PretrainedConfig):
    model_type = "end2end"

    def __init__(self,
                 fnd_clf_hidden_size_1=768,
                 fnd_clf_hidden_size_2=512,
                 fnd_loss_fun=LossFunction.BCE,
                 fnd_class_weights=[1.0, 1.0],
                 fnd_clf_dropout=0.1,
                 lnk_clf_hidden_size_1=768,
                 lnk_clf_hidden_size_2=512,
                 lnk_loss_fun=LossFunction.BCE,
                 lnk_class_weights=[1.0, 1.0],
                 lnk_clf_dropout=0.1,
                 loss_fun=LossFunction.BCE,
                 class_weights=[1.0, 1.0],
                 fnd_one_line_text=False,
                 lnk_one_line_text=False,
                 use_cwe=False,
                 concat_test_first=True,
                 as_javadoc=False,
                 merge_embeddings=False,
                 archi_style=E2EArchitectureStyle.FINDER_LINKER_META,
                 suspect_threshold=0.5,
                 **kwargs):
        self.fnd_clf_hidden_size_1 = fnd_clf_hidden_size_1
        self.fnd_clf_hidden_size_2 = fnd_clf_hidden_size_2
        self.fnd_clf_dropout = fnd_clf_dropout
        self.fnd_loss_fun = fnd_loss_fun
        self.fnd_class_weights = fnd_class_weights
        self.lnk_clf_hidden_size_1 = lnk_clf_hidden_size_1
        self.lnk_clf_hidden_size_2 = lnk_clf_hidden_size_2
        self.lnk_loss_fun = lnk_loss_fun
        self.lnk_class_weights = lnk_class_weights
        self.lnk_clf_dropout = lnk_clf_dropout
        self.loss_fun = loss_fun
        self.class_weights = class_weights
        self.fnd_one_line_text = fnd_one_line_text
        self.lnk_one_line_text = lnk_one_line_text
        self.use_cwe = use_cwe
        self.concat_test_first = concat_test_first
        self.as_javadoc = as_javadoc
        self.merge_embeddings = merge_embeddings
        self.archi_style = archi_style
        self.suspect_threshold = suspect_threshold
        id2label = kwargs.pop("id2label", {0: E2ELabel.NOT_RELATED.value, 1: E2ELabel.RELATED.value})
        label2id = kwargs.pop("label2id", {E2ELabel.NOT_RELATED.value: 0, E2ELabel.RELATED.value: 1})
        super().__init__(num_labels=2, id2label=id2label, label2id=label2id, **kwargs)


"""
Similar to RobertaForSequenceClassification in https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py to have the freedom to modify the architecture. Inspired by https://github.com/awsm-research/LineVul/blob/30429b44f71b7c2c6d64a0180d0e59a795a4d5e3/linevul/linevul_model.py and in line with the tutorial https://huggingface.co/docs/transformers/custom_models
"""


class NeuralNetworkE2E(PreTrainedModel):
    # Must be there to make the model loading work correctly
    config_class = NeuralNetworkE2EConfig
    supports_gradient_checkpointing = True

    def __init__(self,
                 config: NeuralNetworkE2EConfig,
                 fnd_model_class: Type[NeuralNetworkFinder],
                 lnk_model_class: Type[NeuralNetworkLinker],
                 tokenizer_model_name: str,
                 fnd_model: NeuralNetworkFinder = None,
                 lnk_model: NeuralNetworkLinker = None,
                 cache_dir: str = None):
        super().__init__(config)
        self.config = config
        self.num_labels = self.config.num_labels
        if self.config.archi_style == E2EArchitectureStyle.LINKER_ONLY:
            self.fnd_model: NeuralNetworkFinder = None
        else:
            if fnd_model:
                self.fnd_model = fnd_model
            else:
                fnd_config_args = {
                    "clf_hidden_size_1": config.fnd_clf_hidden_size_1,
                    "clf_hidden_size_2": config.fnd_clf_hidden_size_2,
                    "one_line_text": config.fnd_one_line_text,
                    "loss_fun": config.fnd_loss_fun,
                    "class_weights": config.fnd_class_weights,
                }
                self.fnd_model = fnd_model_class(
                    NeuralNetworkFinderConfig(**{k: v for k, v in fnd_config_args.items() if v is not None}),
                    cache_dir=cache_dir
                )
        if lnk_model:
            self.lnk_model: NeuralNetworkLinker = lnk_model
        else:
            lnk_config_args = {
                "clf_hidden_size_1": config.lnk_clf_hidden_size_1,
                "clf_hidden_size_2": config.lnk_clf_hidden_size_2,
                "loss_fun": config.lnk_loss_fun,
                "class_weights": config.lnk_class_weights,
                "one_line_text": config.lnk_one_line_text,
                "use_cwe": config.use_cwe,
                "concat_test_first": config.concat_test_first,
                "as_javadoc": config.as_javadoc,
                "merge_embeddings": config.merge_embeddings,
            }
            self.lnk_model = lnk_model_class(
                NeuralNetworkLinkerConfig(**{k: v for k, v in lnk_config_args.items() if v is not None}),
                cache_dir=cache_dir
            )
        self.base_model_prefix = self.fnd_model.base_model_prefix if self.fnd_model else self.lnk_model.base_model_prefix
        self._no_split_modules = self.fnd_model._no_split_modules if self.fnd_model else self.lnk_model._no_split_modules
        self._keys_to_ignore_on_load_unexpected = self.fnd_model._keys_to_ignore_on_load_unexpected if self.fnd_model else self.lnk_model._keys_to_ignore_on_load_unexpected
        self._tied_weights_keys = self.fnd_model._tied_weights_keys if self.fnd_model else self.lnk_model._tied_weights_keys

        if self.config.archi_style == E2EArchitectureStyle.FINDER_LINKER_META:
            self.meta_out_layer = torch.nn.Linear(self.fnd_model.num_labels + self.lnk_model.num_labels, self.num_labels)
        elif self.config.archi_style == E2EArchitectureStyle.FINDER_LINKER_FUSE:
            self.fnd_last_embedding = None
            self.fnd_model.clf_head.linear_layer_2.register_forward_hook(lambda module, args, output: setattr(self, 'fnd_last_embedding', output))
            self.lnk_last_embedding = None
            self.lnk_model.clf_head.linear_layer_2.register_forward_hook(lambda module, args, output: setattr(self, 'lnk_last_embedding', output))
            # using an average of FND and LNK dropout temporarily
            self.fuse_head = ClassificationHead(
                in_size=self.fnd_model.clf_head.linear_layer_2.out_features + self.lnk_model.clf_head.linear_layer_2.out_features,
                hidden_size_1=64,
                hidden_size_2=None,
                out_size=self.num_labels,
                dropout_p=mean((self.config.fnd_clf_dropout, self.config.lnk_clf_dropout))
            )
            # self.fuse_head = torch.nn.Linear(self.fnd_model.clf_head.linear_layer_2.out_features + self.lnk_model.clf_head.linear_layer_2.out_features, self.num_labels)
        if config.merge_embeddings:
            # Setting model_input_names is fundamental, otherwise the tokenizer will not provide these fields to the forward() method correctly
            self.model_input_names_for_tokenizer = [FND_INPUT_IDS, LNK_INPUT_IDS_1, LNK_INPUT_IDS_2, FND_ATTENTION_MASK, LNK_ATTENTION_MASK_1, LNK_ATTENTION_MASK_2]
        else:
            self.model_input_names_for_tokenizer = [FND_INPUT_IDS, LNK_INPUT_IDS, FND_ATTENTION_MASK, LNK_ATTENTION_MASK]
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name,
            cache_dir=cache_dir,
            model_input_names=self.model_input_names_for_tokenizer
        )
        if self.config.loss_fun == LossFunction.WBCE:
            self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.config.class_weights))
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self,
                fnd_input_ids: Optional[torch.LongTensor] = None,
                fnd_attention_mask: Optional[torch.LongTensor] = None,
                lnk_input_ids: Optional[torch.LongTensor] = None,
                lnk_attention_mask: Optional[torch.FloatTensor] = None,
                lnk_input_ids_1: Optional[torch.LongTensor] = None,
                lnk_input_ids_2: Optional[torch.LongTensor] = None,
                lnk_attention_mask_1: Optional[torch.FloatTensor] = None,
                lnk_attention_mask_2: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[tuple[torch.Tensor], SequenceClassifierOutputE2E]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        fnd_outputs = None
        lnk_outputs = self.lnk_model(
            input_ids=lnk_input_ids,
            attention_mask=lnk_attention_mask,
            input_ids_1=lnk_input_ids_1,
            input_ids_2=lnk_input_ids_2,
            attention_mask_1=lnk_attention_mask_1,
            attention_mask_2=lnk_attention_mask_2,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.config.archi_style == E2EArchitectureStyle.LINKER_ONLY:
            e2e_logits = lnk_outputs.logits
        else:
            fnd_outputs = self.fnd_model(
                input_ids=fnd_input_ids,
                attention_mask=fnd_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if self.config.archi_style == E2EArchitectureStyle.FINDER_LINKER_MASK:
                fnd_logits = fnd_outputs.logits
                probs = torch.nn.functional.softmax(fnd_logits, dim=-1)
                prob_flagged = probs[:, self.fnd_model.config.label2id[FinderLabel.WITNESSING]]
                # NOTE The suspect_threshold is only used in this style. Should be rethought
                mask = prob_flagged >= self.config.suspect_threshold
                e2e_logits = fnd_logits.clone()
                if e2e_logits[mask].size(dim=0) > 0:
                    e2e_logits[mask] = lnk_outputs.logits[mask]
                else:
                    lnk_outputs = None
            elif self.config.archi_style == E2EArchitectureStyle.FINDER_LINKER_META:
                e2e_logits = self.meta_out_layer(torch.concat([fnd_outputs.logits, lnk_outputs.logits], dim=-1))
            elif self.config.archi_style == E2EArchitectureStyle.FINDER_LINKER_FUSE:
                with torch.autocast(device_type=DEVICE.type):
                    e2e_logits = self.fuse_head(torch.concat([self.fnd_last_embedding, self.lnk_last_embedding], dim=-1))

        loss = None
        if labels is not None:
            labels = labels.to(e2e_logits.device)
            loss = self.loss_fct(e2e_logits.view(-1, self.config.num_labels), labels.view(-1))
        if not return_dict:
            output = (e2e_logits,)
            if fnd_outputs:
                output += fnd_outputs[2:]
            if lnk_outputs:
                output += lnk_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutputE2E(
            loss=loss,
            logits=e2e_logits,
            fnd_hidden_states=fnd_outputs.hidden_states if fnd_outputs else None,
            fnd_attentions=fnd_outputs.attentions if fnd_outputs else None,
            lnk_hidden_states=lnk_outputs.hidden_states if lnk_outputs else None,
            lnk_attentions=lnk_outputs.attentions if lnk_outputs else None,
            lnk_hidden_states_1=lnk_outputs.hidden_states_1 if lnk_outputs else None,
            lnk_attentions_1=lnk_outputs.attentions_1 if lnk_outputs else None,
            lnk_hidden_states_2=lnk_outputs.hidden_states_2 if lnk_outputs else None,
            lnk_attentions_2=lnk_outputs.attentions_2 if lnk_outputs else None
        )

    def must_one_line_text(self) -> bool:
        return self.config.lnk_one_line_text

    def must_use_cwe(self) -> bool:
        return self.config.use_cwe

    def must_concat_test_first(self) -> bool:
        return self.config.concat_test_first

    def must_place_text_as_javadoc(self) -> bool:
        return self.config.as_javadoc

    def must_merge_lnk_embeddings(self) -> bool:
        return self.config.merge_embeddings

    def _call_tokenizer(self,
                        text_1: Union[str, List[str], List[List[str]]],
                        text_2: Union[str, List[str], List[List[str]]] = None,
                        truncate: bool = True):
        return call_tokenizer_standard(self.tokenizer, text_1, text_2, truncate)

    def encode_batch(self, instance_batch, truncate: bool = True, **kwargs):
        if self.must_use_cwe():
            instance_batch[TEXT_2_COL] = [prepend_text(text_2, text_2_1) for text_2, text_2_1 in zip(instance_batch[TEXT_2_COL], instance_batch[TEXT_2_1_COL])]
        if self.must_one_line_text():
            instance_batch[TEXT_1_COL] = [one_line_text(inst) for inst in instance_batch[TEXT_1_COL]]
            instance_batch[TEXT_2_COL] = [one_line_text(inst) for inst in instance_batch[TEXT_2_COL]]
        tokenized_fnd = self._call_tokenizer(text_1=instance_batch[TEXT_1_COL], truncate=truncate)
        if self.must_merge_lnk_embeddings():
            learn_1 = self._call_tokenizer(text_1=instance_batch[TEXT_1_COL], truncate=truncate)
            learn_2 = self._call_tokenizer(text_1=instance_batch[TEXT_2_COL], truncate=truncate)
            lnk_part = {
                LNK_INPUT_IDS_1: learn_1[INPUT_IDS],
                LNK_INPUT_IDS_2: learn_2[INPUT_IDS],
                LNK_ATTENTION_MASK_1: learn_1[ATTENTION_MASK],
                LNK_ATTENTION_MASK_2: learn_2[ATTENTION_MASK],
            }
        elif self.must_place_text_as_javadoc():
            text_batch = [
                (JAVADOC_ONE_LINE if self.must_one_line_text() else JAVADOC_MULTILINE).format(v_desc, test_code)
                for test_code, v_desc in zip(instance_batch[TEXT_1_COL], instance_batch[TEXT_2_COL])
            ]
            tokenized_lnk = self._call_tokenizer(text_1=text_batch, truncate=truncate)
            lnk_part = {
                LNK_INPUT_IDS: tokenized_lnk[INPUT_IDS],
                LNK_ATTENTION_MASK: tokenized_lnk[ATTENTION_MASK]
            }
        else:
            left = instance_batch[TEXT_1_COL] if self.must_concat_test_first() else instance_batch[TEXT_2_COL]
            right = instance_batch[TEXT_2_COL] if self.must_concat_test_first() else instance_batch[TEXT_1_COL]
            tokenized_lnk = self._call_tokenizer(text_1=left, text_2=right, truncate=truncate)
            lnk_part = {
                LNK_INPUT_IDS: tokenized_lnk[INPUT_IDS],
                LNK_ATTENTION_MASK: tokenized_lnk[ATTENTION_MASK]
            }
        return {
            FND_INPUT_IDS: tokenized_fnd[INPUT_IDS],
            FND_ATTENTION_MASK: tokenized_fnd[ATTENTION_MASK],
            **lnk_part
        }

    def encode_single(self, test_code: str, vuln: str, truncate: bool = True, **kwargs):
        ready_test_code = one_line_text(test_code) if self.must_one_line_text() else test_code
        ready_vuln = one_line_text(vuln) if self.must_one_line_text() else vuln
        tokenized_fnd = self._call_tokenizer(text_1=ready_test_code, truncate=truncate)
        if self.must_merge_lnk_embeddings():
            learn_1 = self._call_tokenizer(text_1=ready_test_code, truncate=truncate)
            learn_2 = self._call_tokenizer(text_1=ready_vuln, truncate=truncate)
            lnk_part = {
                LNK_INPUT_IDS_1: learn_1[INPUT_IDS],
                LNK_INPUT_IDS_2: learn_2[INPUT_IDS],
                LNK_ATTENTION_MASK_1: learn_1[ATTENTION_MASK],
                LNK_ATTENTION_MASK_2: learn_2[ATTENTION_MASK],
            }
        elif self.must_place_text_as_javadoc():
            ready_test_and_comment = (JAVADOC_ONE_LINE if self.must_one_line_text() else JAVADOC_MULTILINE).format(ready_vuln, ready_test_code)
            tokenized_lnk = self._call_tokenizer(text_1=ready_test_and_comment, truncate=truncate)
            lnk_part = {
                LNK_INPUT_IDS: tokenized_lnk[INPUT_IDS],
                LNK_ATTENTION_MASK: tokenized_lnk[ATTENTION_MASK]
            }
        else:
            left = ready_test_code if self.must_concat_test_first() else ready_vuln
            right = ready_vuln if self.must_concat_test_first() else ready_test_code
            tokenized_lnk = self._call_tokenizer(text_1=left, text_2=right, truncate=truncate)
            lnk_part = {
                LNK_INPUT_IDS: tokenized_lnk[INPUT_IDS],
                LNK_ATTENTION_MASK: tokenized_lnk[ATTENTION_MASK]
            }
        return {
            FND_INPUT_IDS: tokenized_fnd[INPUT_IDS],
            FND_ATTENTION_MASK: tokenized_fnd[ATTENTION_MASK],
            **lnk_part
        }

    def get_relation_score(self, test_code: str, vuln: str) -> float:
        model_input = self.encode_single(test_code, vuln)
        self.eval()
        with torch.inference_mode():
            model_input = {n: model_input[n].to(DEVICE) if n in model_input else None for n in self.tokenizer.model_input_names}
            # NOTE: Hotfix for UnixCoder, as its encode_single returns tensors of [512], instead of [1,512] 
            model_input = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in model_input.items()}
            model_output = self(**model_input)
            logits: torch.Tensor = model_output.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        related_prob = float(probs[0, self.config.label2id[E2ELabel.RELATED]].item())
        return related_prob

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None and not os.path.exists(os.path.join(save_directory, "tokenizer.json")):
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs) -> 'NeuralNetworkE2E':
        fnd_dir = kwargs.get("fnd_dir", None)
        lnk_dir = kwargs.get("lnk_dir", None)
        fnd_model = AutoModel.from_pretrained(fnd_dir) if fnd_dir else None
        lnk_model = AutoModel.from_pretrained(lnk_dir) if lnk_dir else None
        return load_model_with_tokenizer(pretrained_model_name_or_path, super().from_pretrained, *model_args, fnd_model=fnd_model, lnk_model=lnk_model, **kwargs)


class CodeBertE2EModel(NeuralNetworkE2E):
    FND_MODEL_CLASS = CodeBertFinder
    LNK_MODEL_CLASS = CodeBertLinker

    def __init__(self, config: NeuralNetworkE2EConfig,
                 fnd_model: NeuralNetworkFinder = None,
                 lnk_model: NeuralNetworkLinker = None,
                 cache_dir: str = None):
        super().__init__(config,
                         fnd_model_class=self.FND_MODEL_CLASS,
                         lnk_model_class=self.LNK_MODEL_CLASS,
                         tokenizer_model_name="microsoft/codebert-base",
                         fnd_model=fnd_model,
                         lnk_model=lnk_model,
                         cache_dir=cache_dir)
        self.post_init()


class CodeT5PlusE2EModel(NeuralNetworkE2E):
    FND_MODEL_CLASS = CodeT5PlusFinder
    LNK_MODEL_CLASS = CodeT5PlusLinker

    def __init__(self, config: NeuralNetworkE2EConfig,
                 fnd_model: NeuralNetworkFinder = None,
                 lnk_model: NeuralNetworkLinker = None,
                 cache_dir: str = None):
        super().__init__(config,
                         fnd_model_class=self.FND_MODEL_CLASS,
                         lnk_model_class=self.LNK_MODEL_CLASS,
                         tokenizer_model_name="Salesforce/codet5p-220m",
                         fnd_model=fnd_model,
                         lnk_model=lnk_model,
                         cache_dir=cache_dir)
        self.post_init()


class UniXCoderE2EModel(NeuralNetworkE2E):
    FND_MODEL_CLASS = UniXCoderFinder
    LNK_MODEL_CLASS = UniXCoderLinker

    def __init__(self, config: NeuralNetworkE2EConfig,
                 fnd_model: NeuralNetworkFinder = None,
                 lnk_model: NeuralNetworkLinker = None,
                 cache_dir: str = None):
        super().__init__(config,
                         fnd_model_class=self.FND_MODEL_CLASS,
                         lnk_model_class=self.LNK_MODEL_CLASS,
                         tokenizer_model_name="microsoft/unixcoder-base",
                         fnd_model=fnd_model,
                         lnk_model=lnk_model,
                         cache_dir=cache_dir)
        self.post_init()

    def _call_tokenizer(self,
                        text_1: Union[str, List[str], List[List[str]]],
                        text_2: Union[str, List[str], List[List[str]]] = None,
                        truncate: bool = True):
        # Override as UniXCoder does something slightly different, we use this custom helper function
        return call_tokenizer_unixcoder(self.tokenizer, text_1, text_2, truncate)
