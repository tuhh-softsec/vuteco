import json
import os
import sys
from argparse import ArgumentParser

import pandas as pd
from halo import Halo
from vuteco.core.common.cli_args import add_tool_args
from vuteco.core.common.cli_constants import (VUTECO_LM_E2E_TECHNIQUES,
                                              VUTECO_LM_FL_TECHNIQUES,
                                              VUTECO_LM_FND_TECHNIQUES,
                                              VUTECO_NN_E2E_TECHNIQUES,
                                              VUTECO_NN_FL_TECHNIQUES,
                                              VUTECO_NN_FND_TECHNIQUES)
from vuteco.core.common.constants import (REMOTE_BASEPATH,
                                          TRAINED_MODEL_DIRPATH, TechniqueName)
from vuteco.core.common.resources import get_vuln_kb
from vuteco.main.starter import vuteco_start
from vuteco.main.vuteco_domain import (VutecoFixCommit, VutecoGrepFinder,
                                       VutecoGrepMatcher,
                                       VutecoLanguageModelE2E,
                                       VutecoLanguageModelFinder,
                                       VutecoLanguageModelFinderLinker,
                                       VutecoNeuralNetworkE2E,
                                       VutecoNeuralNetworkFinder,
                                       VutecoNeuralNetworkFinderLinker,
                                       VutecoTechnique, VutecoVocabularyFinder)


def prepare_techniques(tech_names: list[str], pretrained_model_name_or_path: str, vuln_matching: bool = True, e2e_enabled: bool = True, local_load: bool = True) -> list[VutecoTechnique]:
    techniques: list[VutecoTechnique] = []
    with Halo(text="Preparing VUTECO") as spinner:
        for tech_name in tech_names:
            if vuln_matching:
                if e2e_enabled:
                    if tech_name in VUTECO_NN_E2E_TECHNIQUES.keys():
                        techniques.append(VutecoNeuralNetworkE2E(tech_name, pretrained_model_name_or_path, local_load))
                    elif tech_name in VUTECO_LM_E2E_TECHNIQUES.keys():
                        techniques.append(VutecoLanguageModelE2E(tech_name, pretrained_model_name_or_path, local_load))
                else:
                    if tech_name in VUTECO_NN_FL_TECHNIQUES.keys():
                        techniques.append(VutecoNeuralNetworkFinderLinker(tech_name, pretrained_model_name_or_path, local_load))
                    elif tech_name in VUTECO_LM_FL_TECHNIQUES.keys():
                        techniques.append(VutecoLanguageModelFinderLinker(tech_name, pretrained_model_name_or_path, local_load))
                if tech_name == TechniqueName.FIX:
                    techniques.append(VutecoFixCommit())
                elif tech_name == TechniqueName.GREP:
                    techniques.append(VutecoGrepMatcher(pretrained_model_name_or_path))
            else:
                if tech_name in VUTECO_NN_FND_TECHNIQUES.keys():
                    techniques.append(VutecoNeuralNetworkFinder(tech_name, pretrained_model_name_or_path, local_load))
                elif tech_name in VUTECO_LM_FND_TECHNIQUES.keys():
                    techniques.append(VutecoLanguageModelFinder(tech_name, pretrained_model_name_or_path, local_load))
                if tech_name == TechniqueName.FIX:
                    techniques.append(VutecoFixCommit())
                elif tech_name == TechniqueName.GREP:
                    techniques.append(VutecoGrepFinder(pretrained_model_name_or_path))
                elif tech_name == TechniqueName.VOCABULARY:
                    techniques.append(VutecoVocabularyFinder(pretrained_model_name_or_path))
        if len(techniques) > 0:
            spinner.succeed("VUTECO prepared!")
        else:
            spinner.fail("No valid technique supplied. VUTECO could not be prepared. Exiting...")
            exit(1)
    print(f"Prepared the following techniques: {','.join([t.name for t in techniques])}")
    return techniques


def main():
    # TODO (Later) Add the support for one project input (i.e., one line of the input file)
    if not sys.stdout.isatty():
        os.environ["TQDM_DISABLE"] = "1"

    argparser = ArgumentParser()
    argparser = add_tool_args(argparser)
    args = argparser.parse_args()

    input_filepath = args.input_file
    if input_filepath is None:
        print(f"File containing the input not found. Exiting.")
        exit(1)
    with open(input_filepath) as fin:
        input_df = pd.read_csv(fin)

    vuln_finding = bool(args.vuln_find)
    vuln_matching = bool(args.vuln_match)
    if not vuln_finding and not vuln_matching:
        print("At least one task between Finding and Matching must be enabled. Exiting.")
        exit(1)
    e2e_enabled = bool(args.e2e)
    local_load: bool
    if args.model_dir:
        model_dirpath = args.model_dir if args.model_dir else TRAINED_MODEL_DIRPATH
        if not os.path.exists(model_dirpath):
            print(f"The indicated local directory with the model weights ({model_dirpath}) does not exist. Exiting.")
            exit(1)
        local_load = True
    else:
        print(f"Attempting to load model weights from Hugging Face remote...")
        model_dirpath = REMOTE_BASEPATH
        local_load = False
    techniques = prepare_techniques(args.techniques.split(","), model_dirpath, vuln_matching=vuln_matching, e2e_enabled=e2e_enabled, local_load=local_load)

    knowledge_base: dict = get_vuln_kb()
    if args.knowledge_base:
        with open(args.knowledge_base) as fin:
            extra_kb = json.load(fin)
        knowledge_base.update(extra_kb)

    output_dirpath = args.output_dir
    revision_style = args.revision
    include_cwe = bool(args.include_cwe)
    batched_inference = bool(args.batched_inference)
    skip_inspected_projects = args.skip_inspected_projects

    vuteco_start(input_df, output_dirpath, techniques, knowledge_base, revision_style,
                 vuln_matching=vuln_matching, vuln_finding=vuln_finding,
                 include_cwe=include_cwe, batched_inference=batched_inference,
                 skip_inspected_projects=skip_inspected_projects)


if __name__ == "__main__":
    main()
