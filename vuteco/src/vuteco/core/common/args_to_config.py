from argparse import Namespace
from enum import Enum
from typing import Optional, Type, Union

from vuteco.core.common.augment_data import AugmentationApproach
from vuteco.core.common.constants import (CODEBERT_FULL, CODET5PLUS_FULL,
                                          UNIXCODER_FULL, E2EArchitectureStyle,
                                          E2ETrainingType, LossFunction,
                                          MergeStyle)
from vuteco.core.modeling.modeling_grep_fnd import GrepFinderConfigKeys
from vuteco.core.modeling.modeling_grep_mtc import GrepMatcherConfigKeys
from vuteco.core.modeling.modeling_lm_e2e import LanguageModelE2EConfigKeys
from vuteco.core.modeling.modeling_lm_fnd import LanguageModelFinderConfigKeys
from vuteco.core.modeling.modeling_lm_lnk import LanguageModelLinkerConfigKeys
from vuteco.core.modeling.modeling_nn_e2e import NeuralNetworkE2EConfigKeys
from vuteco.core.modeling.modeling_nn_fnd import NeuralNetworkFinderConfigKeys
from vuteco.core.modeling.modeling_nn_lnk import NeuralNetworkLinkerConfigKeys
from vuteco.core.modeling.modeling_sim_mtc import SimilarityMatcherConfigKeys
from vuteco.core.modeling.modeling_terms_mtc import TermsMatcherConfigKeys
from vuteco.core.modeling.modeling_vocab_fnd import VocabularyFinderConfigKeys


def parse_augments(augment_arg: str) -> tuple[list[AugmentationApproach], dict[str, list]]:
    if augment_arg is None:
        return [], {}
    techniques = []
    extents = {}
    for an_aug in augment_arg.split(","):
        if an_aug == AugmentationApproach.NONE:
            if an_aug not in techniques:
                techniques.append(an_aug)
            if an_aug not in extents:
                extents[an_aug] = [None]
        elif "-" in an_aug:
            aug_tech, aug_ext = an_aug.split("-")
            if aug_tech == AugmentationApproach.JT or aug_tech == AugmentationApproach.SPAT:
                if aug_tech not in techniques:
                    techniques.append(aug_tech)
                if aug_tech not in extents:
                    extents[aug_tech] = []
                if aug_ext not in extents[aug_tech]:
                    extents[aug_tech].append(int(aug_ext) if aug_ext else 1)
            elif aug_tech == AugmentationApproach.BOOTSTRAP:
                if aug_tech not in techniques:
                    techniques.append(aug_tech)
                if aug_tech not in extents:
                    extents[aug_tech] = []
                if aug_ext not in extents[aug_tech]:
                    extents[aug_tech].append(float(aug_ext) / 100 if aug_ext else 1.0)
            elif aug_tech == AugmentationApproach.NONE:
                if aug_tech not in techniques:
                    techniques.append(aug_tech)
    if len(techniques) == 0:
        return [], {}
    return techniques, extents


def parse_list_of_strings(arg: str, string_type: Type[Enum], default_value=None):
    strings = []
    if arg is None:
        strings = [default_value]
    else:
        strings = [a_string for a_string in arg.split(",") if any((v.value for v in string_type if v.value in a_string))]
        if len(strings) == 0:
            strings = [default_value]
    return strings


def parse_boolean_flags(both_arg: bool, yes_arg: Optional[bool]):
    if both_arg:
        return [True, False]
    return [yes_arg]


def parse_list_of_numbers(numbers: str, num_type, default_value=None):
    return [num_type(e) for e in numbers.split(",")] if numbers is not None else [default_value]


def get_nn_experiment_configs(args: Namespace, key_class: Union[NeuralNetworkFinderConfigKeys, NeuralNetworkLinkerConfigKeys]) -> list[tuple[dict, dict]]:
    configs = []
    techs, exts = parse_augments(args.augment)
    losses = parse_list_of_strings(args.loss, LossFunction, LossFunction.BCE.value)
    one_lines = parse_boolean_flags(args.one_line_both, args.one_line)
    hs1 = parse_list_of_numbers(args.hidden_size_1, int)
    hs2 = parse_list_of_numbers(args.hidden_size_2, int)
    epochs = parse_list_of_numbers(args.epochs, int, default_value=1)
    for t in techs:
        for l in losses:
            for ol in one_lines:
                model_config = {
                    key_class.AUGMENT_TECH: t,
                    key_class.LOSS: l,
                    key_class.ONE_LINE: ol
                }
                hyperparams = {
                    key_class.AUGMENT_EXT: exts[t] if t in exts else [],
                    key_class.HIDDEN_SIZE_1: hs1,
                    key_class.HIDDEN_SIZE_2: hs2,
                    key_class.EPOCHS: epochs
                }
                configs.append((model_config, hyperparams))
    return configs


def get_lm_experiment_configs(args: Namespace, key_class: Union[LanguageModelFinderConfigKeys, LanguageModelLinkerConfigKeys]) -> list[tuple[dict, dict]]:
    configs = []
    unsloth_training = parse_boolean_flags(args.unsloth_training_both, args.unsloth_training)
    techs, exts = parse_augments(args.augment)
    #  losses = parse_list_of_strings(args.loss, LossFunction, LossFunction.BCE.value)
    epochs = parse_list_of_numbers(args.epochs, int, default_value=1)
    for ut in unsloth_training:
        for t in techs:
            # for l in losses:
            model_config = {
                key_class.UNSLOTH_TRAINING: ut,
                key_class.AUGMENT_TECH: t,
                # key_class.LOSS: l,
            }
            hyperparams = {
                key_class.AUGMENT_EXT: exts[t] if t in exts else [],
                key_class.EPOCHS: epochs
            }
            configs.append((model_config, hyperparams))
    return configs


def get_nn_fnd_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    return get_nn_experiment_configs(args, NeuralNetworkFinderConfigKeys)


def get_lm_fnd_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    return get_lm_experiment_configs(args, LanguageModelFinderConfigKeys)


def get_nn_lnk_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    merges = parse_list_of_strings(args.merge, MergeStyle, MergeStyle.CONCAT_TEST_DESCR.value)
    use_cwes = parse_boolean_flags(args.use_cwe_both, args.use_cwe)
    for m in merges:
        for uc in use_cwes:
            sub_configs = get_nn_experiment_configs(args, NeuralNetworkLinkerConfigKeys)
            for sc, hp in sub_configs:
                model_config = {**{
                    NeuralNetworkLinkerConfigKeys.MERGE: m,
                    NeuralNetworkLinkerConfigKeys.USE_CWE: uc
                }, **sc}
                configs.append((model_config, hp))
    return configs


def get_lm_lnk_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    use_cwes = parse_boolean_flags(args.use_cwe_both, args.use_cwe)
    for uc in use_cwes:
        sub_configs = get_lm_experiment_configs(args, LanguageModelLinkerConfigKeys)
        for sc, hp in sub_configs:
            model_config = {**{
                LanguageModelLinkerConfigKeys.USE_CWE: uc
            }, **sc}
            configs.append((model_config, hp))
    return configs


def get_nn_e2e_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    sus_thres = parse_list_of_numbers(args.sus_threshold, float)
    archi_styles = parse_list_of_strings(args.archi_style, E2EArchitectureStyle, E2EArchitectureStyle.FINDER_LINKER_META.value)
    train_types = parse_list_of_strings(args.train_type, E2ETrainingType, E2ETrainingType.PRETRAIN_FINETUNE.value)
    ft_techs, ft_exts = parse_augments(args.ft_augment)
    merges = parse_list_of_strings(args.merge, MergeStyle, MergeStyle.JAVADOC.value)
    use_cwes = parse_boolean_flags(args.use_cwe_both, args.use_cwe)
    fnd_techs, fnd_exts = parse_augments(args.fnd_augment)
    lnk_techs, lnk_exts = parse_augments(args.lnk_augment)
    ft_losses = parse_list_of_strings(args.loss, LossFunction, LossFunction.BCE.value)
    fnd_losses = parse_list_of_strings(args.fnd_loss, LossFunction)
    lnk_losses = parse_list_of_strings(args.lnk_loss, LossFunction)
    fnd_one_lines = parse_boolean_flags(args.fnd_one_line_both, args.fnd_one_line)
    lnk_one_lines = parse_boolean_flags(args.lnk_one_line_both, args.lnk_one_line)
    fnd_hs1 = parse_list_of_numbers(args.fnd_hidden_size_1, int)
    fnd_hs2 = parse_list_of_numbers(args.fnd_hidden_size_2, int)
    fnd_epochs = parse_list_of_numbers(args.fnd_epochs, int, default_value=1)
    lnk_hs1 = parse_list_of_numbers(args.lnk_hidden_size_1, int)
    lnk_hs2 = parse_list_of_numbers(args.lnk_hidden_size_2, int)
    lnk_epochs = parse_list_of_numbers(args.lnk_epochs, int, default_value=1)
    ft_epochs = parse_list_of_numbers(args.ft_epochs, int, default_value=1)
    for st in sus_thres:
        for a_style in archi_styles:
            for a_type in train_types:
                #iter_ft_techs = [AugmentationApproach.NONE.value] if a_type in [E2ETrainingType.PRETRAIN_ONLY] else ft_techs
                for ft_tech in ft_techs:
                    for ft_loss in ft_losses:
                        for m in merges:
                            for uc in use_cwes:
                                if fnd_techs == []:
                                    actual_fnd_techs = [ft_tech]
                                    actual_fnd_exts = {ft_tech: ft_exts[ft_tech]}
                                else:
                                    actual_fnd_techs = fnd_techs.copy()
                                    actual_fnd_exts = fnd_exts.copy()
                                for fnd_tech in actual_fnd_techs:
                                    for fnd_loss in fnd_losses:
                                        if fnd_loss is None:
                                            fnd_loss = ft_loss
                                        if lnk_techs == []:
                                            actual_lnk_techs = [ft_tech]
                                            actual_lnk_exts = {ft_tech: ft_exts[ft_tech]}
                                        else:
                                            actual_lnk_techs = lnk_techs.copy()
                                            actual_lnk_exts = lnk_exts.copy()
                                        for lnk_tech in actual_lnk_techs:
                                            for lnk_loss in lnk_losses:
                                                if lnk_loss is None:
                                                    lnk_loss = ft_loss
                                                for fol in fnd_one_lines:
                                                    for lol in lnk_one_lines:
                                                        model_config = {
                                                            NeuralNetworkE2EConfigKeys.SUSPECT_THRESHOLD: st,
                                                            NeuralNetworkE2EConfigKeys.ARCHI_STYLE: a_style,
                                                            NeuralNetworkE2EConfigKeys.TRAIN_TYPE: a_type,
                                                            NeuralNetworkE2EConfigKeys.FT_AUGMENT_TECH: ft_tech,
                                                            NeuralNetworkE2EConfigKeys.MERGE: m,
                                                            NeuralNetworkE2EConfigKeys.USE_CWE: uc,
                                                            NeuralNetworkE2EConfigKeys.FND_AUGMENT_TECH: fnd_tech,
                                                            NeuralNetworkE2EConfigKeys.FND_LOSS: fnd_loss,
                                                            NeuralNetworkE2EConfigKeys.LNK_AUGMENT_TECH: lnk_tech,
                                                            NeuralNetworkE2EConfigKeys.LNK_LOSS: lnk_loss,
                                                            NeuralNetworkE2EConfigKeys.LOSS: ft_loss,
                                                            NeuralNetworkE2EConfigKeys.FND_ONE_LINE: fol,
                                                            NeuralNetworkE2EConfigKeys.LNK_ONE_LINE: lol,
                                                        }
                                                        hyperparams = {
                                                            NeuralNetworkE2EConfigKeys.FND_AUGMENT_EXT: actual_fnd_exts.get(fnd_tech, []),
                                                            NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_1: fnd_hs1,
                                                            NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_2: fnd_hs2,
                                                            NeuralNetworkE2EConfigKeys.FND_EPOCHS: fnd_epochs,
                                                            NeuralNetworkE2EConfigKeys.LNK_AUGMENT_EXT: actual_lnk_exts.get(lnk_tech, []),
                                                            NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_1: lnk_hs1,
                                                            NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_2: lnk_hs2,
                                                            NeuralNetworkE2EConfigKeys.LNK_EPOCHS: lnk_epochs,
                                                            NeuralNetworkE2EConfigKeys.FT_AUGMENT_EXT: ft_exts.get(ft_tech, []),
                                                            NeuralNetworkE2EConfigKeys.FT_EPOCHS: ft_epochs,
                                                        }
                                                        configs.append((model_config, hyperparams))
    return configs


def get_lm_e2e_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    unsloth_training = parse_boolean_flags(args.unsloth_training_both, args.unsloth_training)
    archi_styles = parse_list_of_strings(args.archi_style, E2EArchitectureStyle, E2EArchitectureStyle.FINDER_LINKER_META.value)
    train_types = parse_list_of_strings(args.train_type, E2ETrainingType, E2ETrainingType.PRETRAIN_FINETUNE.value)
    ft_techs, ft_exts = parse_augments(args.ft_augment)
    use_cwes = parse_boolean_flags(args.use_cwe_both, args.use_cwe)
    lnk_techs, lnk_exts = parse_augments(args.lnk_augment)
    lnk_epochs = parse_list_of_numbers(args.lnk_epochs, int, default_value=1)
    ft_epochs = parse_list_of_numbers(args.ft_epochs, int, default_value=1)
    for ut in unsloth_training:
        for a_style in archi_styles:
            for a_type in train_types:
                #iter_ft_techs = [AugmentationApproach.NONE.value] if a_type in [E2ETrainingType.PRETRAIN_ONLY] else ft_techs
                for ft_tech in ft_techs:
                    for uc in use_cwes:
                        if lnk_techs == []:
                            actual_lnk_techs = [ft_tech]
                            actual_lnk_exts = {ft_tech: ft_exts[ft_tech]}
                        else:
                            actual_lnk_techs = lnk_techs.copy()
                            actual_lnk_exts = lnk_exts.copy()
                        for lnk_tech in actual_lnk_techs:
                            model_config = {
                                LanguageModelE2EConfigKeys.UNSLOTH_TRAINING: ut,
                                LanguageModelE2EConfigKeys.ARCHI_STYLE: a_style,
                                LanguageModelE2EConfigKeys.TRAIN_TYPE: a_type,
                                LanguageModelE2EConfigKeys.FT_AUGMENT_TECH: ft_tech,
                                LanguageModelE2EConfigKeys.USE_CWE: uc,
                                LanguageModelE2EConfigKeys.LNK_AUGMENT_TECH: lnk_tech,
                            }
                            hyperparams = {
                                LanguageModelE2EConfigKeys.LNK_AUGMENT_EXT: actual_lnk_exts.get(lnk_tech, []),
                                LanguageModelE2EConfigKeys.LNK_EPOCHS: lnk_epochs,
                                LanguageModelE2EConfigKeys.FT_AUGMENT_EXT: ft_exts.get(ft_tech, []),
                                LanguageModelE2EConfigKeys.FT_EPOCHS: ft_epochs,
                            }
                            configs.append((model_config, hyperparams))
    return configs


def get_vocab_fnd_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    extractor = [e for e in args.vocab_extractor.split(",") if e in ["iden", "yake"]] if args.vocab_extractor is not None else ["iden"]
    matches = parse_list_of_numbers(args.vocab_matches, int, default_value=1)
    separate = parse_boolean_flags(args.separate_both, args.separate_yes)
    nr_keywords = parse_list_of_numbers(args.keywords_nr, int, default_value=20)
    dedup = [e for e in args.keywords_dedup.split(",") if e in ["leve", "jaro", "seqm"]] if args.keywords_dedup is not None else ["seqm"]
    for e in extractor:
        for s in separate:
            for m in matches:
                for nk in nr_keywords:
                    for d in dedup:
                        model_config = {
                            VocabularyFinderConfigKeys.EXTRACTOR: e,
                            VocabularyFinderConfigKeys.MATCHES: m,
                            VocabularyFinderConfigKeys.SEPARATE: s,
                            SimilarityMatcherConfigKeys.NR_KEYWORDS: nk,
                            SimilarityMatcherConfigKeys.DEDUP: d,
                        }
                        configs.append((model_config, {}))
    return configs


def get_grep_mtc_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    matches = parse_list_of_numbers(args.grep_matches, int, default_value=1)
    extendeds = parse_boolean_flags(args.grep_extended_both, args.grep_extended)
    for m in matches:
        for e in extendeds:
            model_config = {
                GrepMatcherConfigKeys.MATCHES: m,
                GrepMatcherConfigKeys.EXTENDED: e,
            }
            configs.append((model_config, {}))
    return configs


def get_terms_mtc_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    separate = parse_boolean_flags(args.separate_both, args.separate_yes)
    threshold = parse_list_of_numbers(args.terms_threshold, float, default_value=0.5)
    for s in separate:
        for t in threshold:
            model_config = {
                TermsMatcherConfigKeys.SEPARATE: s,
                TermsMatcherConfigKeys.THRESHOLD: t
            }
            configs.append((model_config, {}))
    return configs


def get_sim_mtc_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    extractor = [e for e in args.sim_extractor.split(",") if e in [CODEBERT_FULL, UNIXCODER_FULL, CODET5PLUS_FULL, "yake"]] if args.sim_extractor is not None else [CODEBERT_FULL]
    threshold = parse_list_of_numbers(args.sim_threshold, float, default_value=0.5)
    nr_keywords = parse_list_of_numbers(args.keywords_nr, int, default_value=20)
    dedup = [e for e in args.keywords_dedup.split(",") if e in ["leve", "jaro", "seqm"]] if args.keywords_dedup is not None else ["seqm"]
    for e in extractor:
        for t in threshold:
            for nk in nr_keywords:
                for d in dedup:
                    model_config = {
                        SimilarityMatcherConfigKeys.EXTRACTOR: e,
                        SimilarityMatcherConfigKeys.THRESHOLD: t,
                        SimilarityMatcherConfigKeys.NR_KEYWORDS: nk,
                        SimilarityMatcherConfigKeys.DEDUP: d,
                    }
                    configs.append((model_config, {}))
    return configs


def get_grep_fnd_experiment_configs(args: Namespace) -> list[tuple[dict, dict]]:
    configs = []
    matches = parse_list_of_numbers(args.grep_matches, int, default_value=1)
    extendeds = parse_boolean_flags(args.grep_extended_both, args.grep_extended)
    for m in matches:
        for e in extendeds:
            model_config = {
                GrepFinderConfigKeys.MATCHES: m,
                GrepFinderConfigKeys.EXTENDED: e,
            }
            configs.append((model_config, {}))
    return configs


def parse_augment(augment_arg: str) -> tuple[AugmentationApproach, Union[float, int, None]]:
    if augment_arg is None:
        return AugmentationApproach.NONE, None
    if augment_arg == AugmentationApproach.NONE:
        return AugmentationApproach.NONE, None
    if "-" in augment_arg:
        aug_tech, aug_intens = augment_arg.split("-")
        if aug_tech == AugmentationApproach.JT or aug_tech == AugmentationApproach.SPAT:
            return aug_tech, int(aug_intens) if aug_intens else 1
        elif aug_tech == AugmentationApproach.BOOTSTRAP:
            return aug_tech, float(aug_intens) / 100 if aug_intens else 1.0
        elif aug_tech == AugmentationApproach.NONE:
            return AugmentationApproach.NONE, None
    return AugmentationApproach.NONE, None


def parse_string(arg: str, string_type: Type[Enum], default_value=None):
    if arg is None:
        the_string = default_value
    else:
        if any((v.value for v in string_type if v.value == arg)):
            the_string = arg
        else:
            the_string = default_value
    return the_string


def get_single_nn_config(args: Namespace, key_class: Union[NeuralNetworkFinderConfigKeys, NeuralNetworkLinkerConfigKeys]) -> tuple[dict, dict]:
    tech, ext = parse_augment(args.augment)
    loss = parse_string(args.loss, LossFunction, LossFunction.BCE.value)
    model_config = {
        key_class.AUGMENT_TECH: tech,
        key_class.LOSS: loss,
        key_class.ONE_LINE: args.one_line
    }
    hyperparams = {
        key_class.AUGMENT_EXT: ext,
        key_class.HIDDEN_SIZE_1: int(args.hidden_size_1) if args.hidden_size_1 is not None else None,
        key_class.HIDDEN_SIZE_2: int(args.hidden_size_2) if args.hidden_size_2 is not None else None,
        key_class.EPOCHS: int(args.epochs) if args.epochs is not None else 1,
    }
    return model_config, hyperparams


def get_single_lm_config(args: Namespace, key_class: Union[LanguageModelFinderConfigKeys, LanguageModelLinkerConfigKeys]) -> tuple[dict, dict]:
    tech, ext = parse_augment(args.augment)
    model_config = {
        key_class.UNSLOTH_TRAINING: args.unsloth_training,
        key_class.AUGMENT_TECH: tech,
    }
    hyperparams = {
        key_class.AUGMENT_EXT: ext,
        key_class.EPOCHS: int(args.epochs) if args.epochs is not None else 1,
    }
    return model_config, hyperparams


def get_nn_fnd_config(args: Namespace) -> tuple[dict, dict]:
    return get_single_nn_config(args, NeuralNetworkFinderConfigKeys)


def get_lm_fnd_config(args: Namespace) -> tuple[dict, dict]:
    return get_single_lm_config(args, LanguageModelFinderConfigKeys)


def get_nn_lnk_config(args: Namespace) -> tuple[dict, dict]:
    base_model_config, hyperparams = get_single_nn_config(args, NeuralNetworkLinkerConfigKeys)
    model_config = {**{
        NeuralNetworkLinkerConfigKeys.MERGE: parse_string(args.merge, MergeStyle, MergeStyle.CONCAT_TEST_DESCR.value),
        NeuralNetworkLinkerConfigKeys.USE_CWE: args.use_cwe
    }, **base_model_config}
    return model_config, hyperparams


def get_lm_lnk_config(args: Namespace) -> tuple[dict, dict]:
    base_model_config, hyperparams = get_single_lm_config(args, LanguageModelLinkerConfigKeys)
    model_config = {**{
        LanguageModelLinkerConfigKeys.USE_CWE: args.use_cwe
    }, **base_model_config}
    return model_config, hyperparams


def get_nn_e2e_config(args: Namespace) -> tuple[dict, dict]:
    ft_tech, ft_ext = parse_augment(args.ft_augment)
    fnd_tech, fnd_ext = parse_augment(args.fnd_augment)
    lnk_tech, lnk_ext = parse_augment(args.lnk_augment)
    fnd_loss = parse_string(args.fnd_loss, LossFunction, LossFunction.BCE.value)
    lnk_loss = parse_string(args.lnk_loss, LossFunction, LossFunction.BCE.value)
    loss = parse_string(args.loss, LossFunction, LossFunction.BCE.value)
    model_config = {
        NeuralNetworkE2EConfigKeys.SUSPECT_THRESHOLD: float(args.sus_threshold) if args.sus_threshold is not None else None,
        NeuralNetworkE2EConfigKeys.ARCHI_STYLE: parse_string(args.archi_style, E2EArchitectureStyle, E2EArchitectureStyle.FINDER_LINKER_META.value),
        NeuralNetworkE2EConfigKeys.TRAIN_TYPE: parse_string(args.train_type, E2ETrainingType, E2ETrainingType.PRETRAIN_FINETUNE.value),
        NeuralNetworkE2EConfigKeys.FT_AUGMENT_TECH: ft_tech,
        NeuralNetworkE2EConfigKeys.MERGE: parse_string(args.merge, MergeStyle, MergeStyle.CONCAT_TEST_DESCR.value),
        NeuralNetworkE2EConfigKeys.USE_CWE: args.use_cwe,
        NeuralNetworkE2EConfigKeys.FND_AUGMENT_TECH: fnd_tech,
        NeuralNetworkE2EConfigKeys.FND_LOSS: fnd_loss,
        NeuralNetworkE2EConfigKeys.LNK_AUGMENT_TECH: lnk_tech,
        NeuralNetworkE2EConfigKeys.LNK_LOSS: lnk_loss,
        NeuralNetworkE2EConfigKeys.LOSS: loss,
        NeuralNetworkE2EConfigKeys.FND_ONE_LINE: args.fnd_one_line,
        NeuralNetworkE2EConfigKeys.LNK_ONE_LINE: args.fnd_one_line,
    }
    hyperparams = {
        NeuralNetworkE2EConfigKeys.FND_AUGMENT_EXT: fnd_ext,
        NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_1: int(args.fnd_hidden_size_1) if args.fnd_hidden_size_1 is not None else None,
        NeuralNetworkE2EConfigKeys.FND_HIDDEN_SIZE_2: int(args.fnd_hidden_size_2) if args.fnd_hidden_size_2 is not None else None,
        NeuralNetworkE2EConfigKeys.FND_EPOCHS: int(args.fnd_epochs) if args.fnd_epochs is not None else 1,
        NeuralNetworkE2EConfigKeys.LNK_AUGMENT_EXT: lnk_ext,
        NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_1: int(args.lnk_hidden_size_1) if args.lnk_hidden_size_1 is not None else None,
        NeuralNetworkE2EConfigKeys.LNK_HIDDEN_SIZE_2: int(args.lnk_hidden_size_2) if args.lnk_hidden_size_2 is not None else None,
        NeuralNetworkE2EConfigKeys.LNK_EPOCHS: int(args.lnk_epochs) if args.lnk_epochs is not None else 1,
        NeuralNetworkE2EConfigKeys.FT_AUGMENT_EXT: ft_ext,
        NeuralNetworkE2EConfigKeys.FT_EPOCHS: int(args.ft_epochs) if args.ft_epochs is not None else 1,
    }
    return model_config, hyperparams


def get_lm_e2e_config(args: Namespace) -> tuple[dict, dict]:
    ft_tech, ft_ext = parse_augment(args.ft_augment)
    lnk_tech, lnk_ext = parse_augment(args.lnk_augment)
    model_config = {
        LanguageModelE2EConfigKeys.UNSLOTH_TRAINING: args.unsloth_training,
        LanguageModelE2EConfigKeys.ARCHI_STYLE: parse_string(args.archi_style, E2EArchitectureStyle, E2EArchitectureStyle.FINDER_LINKER_META.value),
        LanguageModelE2EConfigKeys.TRAIN_TYPE: parse_string(args.train_type, E2ETrainingType, E2ETrainingType.PRETRAIN_FINETUNE.value),
        LanguageModelE2EConfigKeys.FT_AUGMENT_TECH: ft_tech,
        LanguageModelE2EConfigKeys.USE_CWE: args.use_cwe,
        LanguageModelE2EConfigKeys.LNK_AUGMENT_TECH: lnk_tech,
    }
    hyperparams = {
        LanguageModelE2EConfigKeys.LNK_AUGMENT_EXT: lnk_ext,
        LanguageModelE2EConfigKeys.LNK_EPOCHS: int(args.lnk_epochs) if args.lnk_epochs is not None else 1,
        LanguageModelE2EConfigKeys.FT_AUGMENT_EXT: ft_ext,
        LanguageModelE2EConfigKeys.FT_EPOCHS: int(args.ft_epochs) if args.ft_epochs is not None else 1,
    }
    return model_config, hyperparams
