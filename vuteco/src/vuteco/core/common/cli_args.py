import datetime as dt
import os
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Callable
from typing import Union

from pandas import DataFrame
from vuteco.core.common.augment_data import AugmentationApproach
from vuteco.core.common.cli_constants import (LM_FINDER_MODELS,
                                              LM_LINKER_MODELS, NN_E2E_MODELS,
                                              NN_FINDER_MODELS,
                                              NN_LINKER_MODELS)
from vuteco.core.common.constants import (DATA_DIRPATH, RAW_FILENAME,
                                          VUL4J_TEST_FILEPATH,
                                          E2EArchitectureStyle,
                                          E2ETrainingType, End2EndName,
                                          FinderName, HeuristicName,
                                          LinkerName, LossFunction, MergeStyle,
                                          TechniqueName, VutecoRevisionStyle)
from vuteco.core.common.utils_training import print_stdout_file

BASE_ARGS = {
    "finder": {
        "names": ["-fnd", "--finder"],
        "props": {
            "choices": [e.value for e in FinderName],
            "help": 'Finder model to train.'
        }
    },
    "linker": {
        "names": ["-lnk", "--linker"],
        "props": {
            "choices": [e.value for e in LinkerName],
            "help": 'Linker model to train.'
        }
    },
    "end-to-end": {
        "names": ["-e2e", "--end-to-end"],
        "props": {
            "choices": [e.value for e in End2EndName],
            "help": 'End-to-End model to train.'
        }
    },
    "heuristic": {
        "names": ["-heu", "--heuristic"],
        "props": {
            "choices": [e.value for e in HeuristicName],
            "help": 'Heuristic to train.'
        }
    },
    "input": {
        "names": ["-i", "--input"],
        "props": {
            "default": DATA_DIRPATH,
            "help": 'Directory containing the Vul4J dataset (vul4j_tests.json file and vul4j_tests directory) used to feed the models (i.e., training).'
        }
    },
    "output": {
        "names": ["-o", "--output"],
        "props": {
            "help": 'Directory where to export the results (i.e., the trained model or the evaluation results).'
        }
    },
    "cache": {
        "names": ["-ca", "--cache"],
        "props": {
            "help": '(Only when using HuggingFace models) Directory for caching HuggingFace data. If not provided, the default one in the home directory will be used'
        },
    },
    "debug": {
        "names": ["--debug"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": 'If enabled, any training process is made on a very small subset of the input dataset, just to ensure the script terminates in much less time than an ordinary run.'
        }
    }
}


CONFIG_ARGS = {
    "splits": {
        "names": ["-s", "--splits"],
        "props": {
            "help": 'Dataset splits to experiment expressed pairs or triples of float numbers summing 1 separated by hyphen (without spaces), e.g., 0.7-0.15-0.15'
        }
    },
    "cv": {
        "names": ["-cv"],
        "props": {
            "help": 'Number of rounds for the Monte Carlo Cross Validation, e.g., 3'
        }
    },
    "metric": {
        "names": ["-m", "--metric"],
        "props": {
            "help": 'Metric to optimize during training, e.g., f1 or auc_roc'
        }
    },
    "archi-style": {
        "names": ["-as", "--archi-style"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Indicate which architecture to use. {E2EArchitectureStyle.LINKER_ONLY} will use only the Linker model for the End-to-end task (i.e., the Finder is disabled).'
        }
    },
    "train-type": {
        "names": ["-tt", "--train-type"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Indicate which type of training to do. {E2ETrainingType.PRETRAIN_ONLY} will run the separate training for Finder and Linker modules. {E2ETrainingType.FINETUNE_ONLY} will only run the fine-tuning of the whole End-to-end model (so, Finder and Linker are loaded with default weights). {E2ETrainingType.PRETRAIN_FINETUNE} will first run the separate training for Finder and Linker modules and then fine-tune the whole End-to-end model.'
        }
    },
    "sus-threshold": {
        "names": ["-st", "--sus-threshold"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Probability threshold for flagging a witnessing test as suspect before calling the Linker model, e.g., 0.3.'
        }
    },
    "epochs": {
        "names": ["-e", "--epochs"],
        "props": {
            "help": f'({", ".join(NN_FINDER_MODELS.keys()) + ", ".join(NN_LINKER_MODELS.keys())}) Number of epochs to train the model, e.g., 3.'
        }
    },
    "fnd-epochs": {
        "names": ["--fnd-epochs"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Number of epochs for pre-training the Finder module, e.g., 3'
        }
    },
    "lnk-epochs": {
        "names": ["--lnk-epochs"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Number of epochs for pre-training the Linker module, e.g., 3'
        }
    },
    "ft-epochs": {
        "names": ["--ft-epochs"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Numbers of epochs for the fine-tuning the whole End-to-End model, e.g., 3'
        }
    },
    "hidden-size-1": {
        "names": ["-hs1", "--hidden-size-1"],
        "props": {
            "help": f'({", ".join(NN_FINDER_MODELS.keys()) + ", ".join(NN_LINKER_MODELS.keys())}) Size of the first hidden layer of the classification head, e.g., 512'
        }
    },
    "fnd-hidden-size-1": {
        "names": ["--fnd-hidden-size-1"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Sizes of the first hidden layer of the Finder module, e.g., 512'
        }
    },
    "lnk-hidden-size-1": {
        "names": ["--lnk-hidden-size-1"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Sizes of the first hidden layer of the Linker module, e.g., 512'
        }
    },
    "hidden-size-2": {
        "names": ["-hs2", "--hidden-size-2"],
        "props": {
            "help": f'({", ".join(NN_FINDER_MODELS.keys()) + ", ".join(NN_LINKER_MODELS.keys())}) Size of the second hidden layer of the classification head, e.g., 256'
        }
    },
    "fnd-hidden-size-2": {
        "names": ["--fnd-hidden-size-2"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Sizes of the second hidden layer of the Finder module, e.g., 512'
        }
    },
    "lnk-hidden-size-2": {
        "names": ["--lnk-hidden-size-2"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Sizes of the second hidden layer of the Linker module, e.g., 512'
        }
    },
    "augment": {
        "names": ["-au", "--augment"],
        "props": {
            "help": f'({", ".join(NN_FINDER_MODELS.keys()) + ", ".join(NN_LINKER_MODELS.keys())}) Data augmentation technique to apply on the training set. The bootstrap ratio for {AugmentationApproach.BOOTSTRAP} can be set with a number between 0 and 100, e.g., {AugmentationApproach.BOOTSTRAP}-50. The number of repetitions for {AugmentationApproach.JT} and {AugmentationApproach.SPAT} can be set with a positive number, e.g., {AugmentationApproach.JT}-10. The special keyword {AugmentationApproach.NONE} will not use any augmentation.'
        }
    },
    "fnd-augment": {
        "names": ["--fnd-augment"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Data augmentation technique to apply on the training set of the Finder module. The bootstrap ratio for {AugmentationApproach.BOOTSTRAP} can be set with a number between 0 and 100, e.g., {AugmentationApproach.BOOTSTRAP}-50. The number of repetitions for {AugmentationApproach.JT} and {AugmentationApproach.SPAT} can be set with a positive number, e.g., {AugmentationApproach.JT}-10. The special keyword {AugmentationApproach.NONE} will not use any augmentation.'
        }
    },
    "lnk-augment": {
        "names": ["--lnk-augment"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Data augmentation technique to apply on the training set of the Linker module (only the test code). The bootstrap ratio for {AugmentationApproach.BOOTSTRAP} can be set with a number between 0 and 100, e.g., {AugmentationApproach.BOOTSTRAP}-50. The number of repetitions for {AugmentationApproach.JT} and {AugmentationApproach.SPAT} can be set with a positive number, e.g., {AugmentationApproach.JT}-10. The special keyword {AugmentationApproach.NONE} will not use any augmentation.'
        }
    },
    "ft-augment": {
        "names": ["--ft-augment"],
        "props": {
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) Data augmentation technique to apply on the training set of the End-to-End model (only the test code). The bootstrap ratio for {AugmentationApproach.BOOTSTRAP} can be set with a number between 0 and 100, e.g., {AugmentationApproach.BOOTSTRAP}-50. The number of repetitions for {AugmentationApproach.JT} and {AugmentationApproach.SPAT} can be set with a positive number, e.g., {AugmentationApproach.JT}-10. The special keyword {AugmentationApproach.NONE} will not use any augmentation.'
        }
    },
    "loss": {
        "names": ["-l", "--loss"],
        "props": {
            "help": f'Loss function to use when training the model. It can be any value among {[e.value for e in LossFunction]}.'
        }
    },
    "fnd-loss": {
        "names": ["-fl", "--fnd-loss"],
        "props": {
            "help": f'Loss function to use when pre-training the Finder module. It can be any value among {[e.value for e in LossFunction]}.'
        }
    },
    "lnk-loss": {
        "names": ["-ll", "--lnk-loss"],
        "props": {
            "help": f'Loss function to use when pre-training the Linker module. It can be any value among {[e.value for e in LossFunction]}.'
        }
    },
    "merge": {
        "names": ["-me", "--merge"],
        "props": {
            "help": f'({", ".join(NN_LINKER_MODELS.keys()) + ", ".join(NN_E2E_MODELS.keys())}) Technique to use to merge the test code and the CVE description. The technique \"{MergeStyle.LEARN}\" will tokenize and embed the two sentences separatedly and then merge them via a new linear layer before the classification head. The technique \"{MergeStyle.CONCAT_TEST_DESCR}\" will concatenate the test code and description (in this order) before the tokenization and embedding, while \"{MergeStyle.CONCAT_DESCR_TEST}\" will do the same in the other way around. The technique \"{MergeStyle.JAVADOC}\" will place the description as a Javadoc comment before the test code.'
        }
    },
    "one-line": {
        "names": ["--one-line"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'({", ".join(NN_FINDER_MODELS.keys()) + ", ".join(NN_LINKER_MODELS.keys()) + ", ".join(NN_E2E_MODELS.keys())}) If enabled, the input text is made into one line. Gives the precedence to --one-line-both.'
        }
    },
    "one-line-both": {
        "names": ["--one-line-both"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'({", ".join(NN_FINDER_MODELS.keys()) + ", ".join(NN_LINKER_MODELS.keys())}) If enabled, two scenarios are tested: the input left as-is and into one line. This has the precedence over --one-line.'
        }
    },
    "fnd-one-line": {
        "names": ["--fnd-one-line"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) If enabled, the input text is made into one line for the Finder module.'
        }
    },
    "fnd-one-line-both": {
        "names": ["--fnd-one-line-both"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) If enabled, two scenarios are tested for the Finder model: the input left as-is and into one line. This has the precedence over --fnd-one-line.'
        }
    },
    "lnk-one-line": {
        "names": ["--lnk-one-line"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) If enabled, the input text is made into one line for the Linker module and the End-to-End task.'
        }
    },
    "lnk-one-line-both": {
        "names": ["--lnk-one-line-both"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'({", ".join(NN_E2E_MODELS.keys())}) If enabled, two scenarios are tested for the Linker model and the End-to-End task: the input left as-is and into one line. This has the precedence over --lnk-one-line.'
        }
    },
    "use-cwe": {
        "names": ["--use-cwe"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'({", ".join(NN_LINKER_MODELS.keys()) + ", ".join(NN_E2E_MODELS.keys())}) If enabled, the CWE name alongside the description is used, if available.'
        }
    },
    "use-cwe-both": {
        "names": ["--use-cwe-both"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'({", ".join(NN_LINKER_MODELS.keys())}) If enabled, the CWE name alongside the description is tested in addition, if available. This has the precedence over --use-cwe.'
        }
    },
    "unsloth-training": {
        "names": ["-ut", "--unsloth-training"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'({", ".join(LM_FINDER_MODELS.keys()) + ", ".join(LM_LINKER_MODELS.keys())}) If enabled, the fine-tuning is made with Unsloth.'
        }
    },
    "unsloth-training-both": {
        "names": ["--unsloth-training-both"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'({", ".join(NN_LINKER_MODELS.keys())}) If enabled, the fine-tuning of language models is made with Unsloth, in addition. This has the precedence over --unsloth-training.'
        }
    },
    "unsloth-training": {
        "names": ["--unsloth-training"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'({", ".join(NN_LINKER_MODELS.keys())}) If enabled, the fine-tuning of language models is made with Unsloth. Gives the precedence to --unsloth-training.'
        }
    },
    "vocab-matches": {
        "names": ["-vm", "--vocab-matches"],
        "props": {
            "help": f'({FinderName.VOCABULARY_FND}) Number of matches with the fitted vocabulary required to flag a test, e.g., 3'
        }
    },
    "vocab-extractor": {
        "names": ["-ve", "--vocab-extractor"],
        "props": {
            "help": f'({FinderName.VOCABULARY_FND}) Algorithm to use to extract the terms from the input, e.g., yake'
        }
    },
    "separate": {
        "names": ["--separate"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'If enabled, the term splitting with "IDEN" while fitting the vocabulary will separate the parts from CamelCase and snake_case identifiers.'
        }
    },
    "separate-both": {
        "names": ["--separate-both"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'If enabled, the term splitting with "IDEN" while fitting the vocabulary will separate the parts from CamelCase and snake_case identifiers in addition. This has the precedence over --separate-yes.'
        }
    },
    "separate-yes": {
        "names": ["--separate-yes"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'If enabled, only the term splitting with "IDEN" while fitting the vocabulary will separate the parts from CamelCase and snake_case identifiers. Gives the precedence to --separate-both'
        }
    },
    "grep-matches": {
        "names": ["-gm", "--grep-matches"],
        "props": {
            "help": f'({FinderName.GREP_FND}, {LinkerName.GREP_LNK}, {End2EndName.GREP_E2E}) Number of matches with the patterns required to flag a test, e.g., 3.'
        }
    },
    "grep-extended-both": {
        "names": ["--grep-extended-both"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'({FinderName.GREP_FND}, {LinkerName.GREP_LNK}, {End2EndName.GREP_E2E}) If enabled, the search will use an extended set of patterns in addition. This has the precedence over --grep-extended.'
        }
    },
    "grep-extended": {
        "names": ["--grep-extended"],
        "props": {
            "action": BooleanOptionalAction,
            "default": None,
            "help": f'({FinderName.GREP_FND}, {LinkerName.GREP_LNK}, {End2EndName.GREP_E2E}) If enabled, the search will use an extended set of patterns. Gives the precedence to --grep-extended-both.'
        }
    },
    "terms-threshold": {
        "names": ["-tet", "--terms-threshold"],
        "props": {
            "help": f'({LinkerName.TERMS_LNK}, {End2EndName.TERMS_E2E}) Ratio of terms in test code that must be in the vulnerability description to validate the link, e.g., 0.5'
        }
    },
    "sim-threshold": {
        "names": ["-sit", "--sim-threshold"],
        "props": {
            "help": f'({LinkerName.SIM_LNK}, {End2EndName.SIM_E2E}) Degree of similarity required between test code and a vulnerability description to confirm the link, e.g., 0.5'
        }
    },
    "sim-extractor": {
        "names": ["-se", "--sim-extractor"],
        "props": {
            "help": f'({LinkerName.SIM_LNK}, {End2EndName.SIM_E2E}) Algorithm to use to extract the terms or embeddings from the input, e.g., yake or codebert'
        }
    },
    "keywords-nr": {
        "names": ["-knr", "--keywords-nr"],
        "props": {
            "help": f'({LinkerName.SIM_LNK}, {End2EndName.SIM_E2E}) Number keywords to extract from test code and vulnerability description with YAKE, e.g., 20'
        }
    },
    "keywords-dedup": {
        "names": ["-kd", "--keywords-dedup"],
        "props": {
            "help": f'({LinkerName.SIM_LNK}, {End2EndName.SIM_E2E}) Algorithm to deduplicate keywords extracted with YAKE in test code and a vulnerability description, e.g., seqm'
        }
    },
}

TOOL_ARGS = {
    "vuln-find": {
        "names": ["--vuln-find"],
        "props": {
            "action": BooleanOptionalAction,
            "default": True,
            "help": f'When used, VUTECO will NOT predict whether the collected tests are vulnerability-witnessing tests. By default, the finding happens.'
        }
    },
    "vuln-match": {
        "names": ["--vuln-match"],
        "props": {
            "action": BooleanOptionalAction,
            "default": True,
            "help": f'When used, VUTECO will NOT match the collected vulnerability-witnessing tests to the right vulnerability (whose description is supplied via input). By default, the matching happens.'
        }
    },
    "e2e": {
        "names": ["--e2e"],
        "props": {
            "action": BooleanOptionalAction,
            "default": True,
            "help": f'When used, VUTECO will match using the combined Finder+Linker models (a.k.a. E2E model). When not enable, VUTECO will invoke Finder and Linker one after another. By default, the two are combined.'
        }
    },
    "batched-inference": {
        "names": ["--batched-inference"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'When used, VUTECO will create batches of test cases (max 512) for deep learning model during inference to speed up. When not enable, one test case at a time will be given as input.'
        }
    },
    "include-cwe": {
        "names": ["--include-cwe"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'When enabled, VUTECO will also append CWE data (if available) in the vulnerability description during linking.'
        }
    },
    "techniques": {
        "names": ["-t", "--techniques"],
        "props": {
            "help": f'Technique to run. Can be one or multiple, separated by commas without spaces. THe list of supported technique is {",".join(t for t in TechniqueName)}. Some techniques can only flag potential vulnerability-witnessing tests and cannot link them to the vulnerabilities.',
            "default": [],
        }
    },
    "input-file": {
        "names": ["-i", "--input-file"],
        "props": {
            "help": 'Input CSV file containing the project revisions to inspect for collecting vulnerability-witnessing tests.'
        }
    },
    "output-dir": {
        "names": ["-o", "--output-dir"],
        "props": {
            "help": 'Destination of the output files.'
        }
    },
    "model-dir": {
        "names": ["--model-dir"],
        "props": {
            "help": 'Location of the trained models to load.'
        }
    },
    "cache": {
        "names": ["-ca", "--cache"],
        "props": {
            "help": '(Only when using HuggingFace models) Directory for caching HuggingFace data. If not provided, the default one in the home directory will be used'
        },
    },
    "revision": {
        "names": ["-r", "--revision"],
        "props": {
            "default": VutecoRevisionStyle.HEAD,
            "help": f'Indicates which revision (commit) to analyze in each project. \"{VutecoRevisionStyle.ALL}\" will inspect all its revisions. \"{VutecoRevisionStyle.HEAD}\" will inspect only the most recent revision. \"{VutecoRevisionStyle.INPUT_FILE}\" will inspect the revision supplied via the input file in a field named \"revision\".'
        },
        "choices": [r for r in VutecoRevisionStyle]
    },
    "inspect-after-patch": {
        "names": ["--inspect-after-patch"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'When enabled, VUTECO will also inspect the revision after all the fixes of a vulnerability when inspecting it in a project. The fixes will be searched in VUTECO\'s knowledge base. If none is found, nothing will be done.'
        }
    },
    "skip-inspected-projects": {
        "names": ["--skip-inspected-projects"],
        "props": {
            "action": BooleanOptionalAction,
            "default": False,
            "help": f'When enabled, VUTECO will not inspect projects previously analyzed (i.e., if the project result JSON file exists in the output directory already).'
        }
    },
    "knowledge-base": {
        "names": ["-kb", "--knowledge-base"],
        "props": {
            "help": f'Path to a JSON file containing additional info about known vulnerabilities that will temporary update the standard knowledge base in VUTECO, that is, adding new entires and overwriting existing ones. The file must have some vulnerability IDs as keys.'
        }
    }
}


def add_args(argparser: ArgumentParser, supported_args: dict, *args_chosen: str):
    if len(args_chosen) > 0:
        args_to_add = args_chosen
    else:
        args_to_add = list(supported_args.keys())
    for arg in args_to_add:
        if arg not in supported_args:
            continue
        if arg not in [action.dest for action in argparser._actions]:
            argparser.add_argument(*supported_args[arg]["names"], **supported_args[arg]["props"])
    return argparser


def add_base_args(argparser: ArgumentParser, *args_chosen: str):
    return add_args(argparser, BASE_ARGS, *args_chosen)


def add_config_args(argparser: ArgumentParser, *args_chosen: str):
    return add_args(argparser, CONFIG_ARGS, *args_chosen)


def add_tool_args(argparser: ArgumentParser, *args_chosen: str):
    return add_args(argparser, TOOL_ARGS, *args_chosen)


def parse_common_args(args: Namespace, default_outdir: str, input_load_fn: Callable[[str, str, int], DataFrame], start_session: bool = True) -> Union[tuple[DataFrame, str, str], tuple[DataFrame, str]]:
    if args.output is None:
        print_stdout_file(f"Directory where to write the output results not supplied. Using the default location {default_outdir}.")
        out_dirpath = os.path.abspath(default_outdir)
    else:
        out_dirpath = os.path.abspath(args.output)

    if start_session:
        session_outdir = os.path.join(out_dirpath, dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(session_outdir, exist_ok=True)
        session_outfile = os.path.join(session_outdir, RAW_FILENAME)
        with open(session_outfile, "w") as _:
            pass
    else:
        session_outdir = None
        session_outfile = None

    if args.input is None:
        print_stdout_file(f"Directory containing the Vul4J dataset not supplied. Using the default expected location in \"{DATA_DIRPATH}\"", session_outfile)
        dataset_basepath = os.path.abspath(DATA_DIRPATH)
    else:
        dataset_basepath = os.path.abspath(args.input)

    print_stdout_file("Loading Vul4J Dataset...", session_outfile)
    vul4j_df = input_load_fn(VUL4J_TEST_FILEPATH, dataset_basepath)
    if vul4j_df is None:
        print_stdout_file("Failed to load Vul4J dataset. Exiting", session_outfile)
        exit(1)

    if start_session:
        return vul4j_df, session_outdir, session_outfile
    else:
        return vul4j_df, out_dirpath
