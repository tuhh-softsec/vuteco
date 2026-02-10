import json
import os
from abc import ABC

import git
from common.cli_constants import (LM_E2E_MODELS, LM_FINDER_MODELS, LM_FL_MODELS,
                           NN_E2E_MODELS, NN_FINDER_MODELS, NN_FL_MODELS,
                           VUTECO_LM_E2E_TECHNIQUES, VUTECO_LM_FL_TECHNIQUES,
                           VUTECO_LM_FND_TECHNIQUES, VUTECO_NN_E2E_TECHNIQUES,
                           VUTECO_NN_FL_TECHNIQUES, VUTECO_NN_FND_TECHNIQUES)
from common.constants import FINAL, End2EndName, FinderName, TechniqueName
from common.utils_mining import chunk_list, make_hash, prepend_text
from modeling.modeling_fix import FixCommitModel
from modeling.modeling_fl import FinderLinkerWrapper
from modeling.modeling_grep_fnd import GrepFinder
from modeling.modeling_grep_mtc import GrepMatcher
from modeling.modeling_vocab_fnd import VocabularyFinder
from tqdm import tqdm


class TestCase():
    repo: str
    file_path: str
    class_name: str
    method_name: str
    code: str
    startline: int

    def __init__(self, repo: str, file_path: str, class_name: str, method_name: str, code: str, startline: int):
        self.repo = repo
        self.file_path = file_path
        self.class_name = class_name
        self.method_name = method_name
        self.code = code
        self.startline = startline

    def __key(self):
        return (self.repo, self.file_path, self.class_name, self.method_name, self.code)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, TestCase):
            return self.__key() == __value.__key()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__key())

    @property
    def id(self) -> str:
        return f"Test_{make_hash(*self.__key())}"

    # @property
    # def __dict__(self):
    #    return {
    #        'repo': self.repo,
    #        'file_path': self.file_path,
    #        "class_name": self.class_name,
    #        "method_name": self.method_name,
    #        "code": self.code
    #    }


class VutecoTechnique(ABC):
    def __init__(self, name: TechniqueName, model):
        self.name = name
        self.model = model

    def __call__(self, tests: list[TestCase], description: str = None, cwes: str = None, *args, **kwargs) -> dict[TestCase, float]:
        # if description is None:
        #    print(f"Going to find witnessing tests using {self.name}")
        # else:
        #    print(f"Going to match witnessing tests and vulnerability using {self.name}")
        batched_inference = kwargs.get("batched_inference", False)
        try:
            if description is None:
                if batched_inference:
                    scores = []
                    for batch in tqdm(chunk_list(tests), desc=f"    - Finding witnessing tests using {self.name}"):
                        scores.extend(self.model.get_witnessing_score([t.code for t in batch]))
                    return {t: s for t, s in zip(tests, scores)}
                else:
                    return {t: self.model.get_witnessing_score(t.code) for t in tqdm(tests, desc=f"    - Finding witnessing tests using {self.name}")}
            else:
                if batched_inference:
                    scores = []
                    for batch in tqdm(chunk_list(tests), desc=f"    - Matching witnessing tests and vulnerability using {self.name}"):
                        scores.extend(self.model.get_relation_score([t.code for t in batch], prepend_text(description, cwes)))
                    return {t: s for t, s in zip(tests, scores)}
                else:
                    return {t: self.model.get_relation_score(t.code, prepend_text(description, cwes)) for t in tqdm(tests, desc=f"    - Matching witnessing tests and vulnerability using {self.name}")}
        except AttributeError:
            print("       - Operation requested not supported. No results can be returned.")
            return {}


class VutecoNeuralNetworkFinder(VutecoTechnique):
    def __init__(self, tech_name: TechniqueName, model_dir: str, local_load: bool = True) -> None:
        model_name = VUTECO_NN_FND_TECHNIQUES[tech_name]
        model_class, _ = NN_FINDER_MODELS[model_name]
        # TODO Improve management of where to export models
        model_path = os.path.join(model_dir, model_name, FINAL) if local_load else model_dir.format(model=model_name)
        super().__init__(tech_name, model_class.from_pretrained(model_path))


class VutecoLanguageModelFinder(VutecoTechnique):
    def __init__(self, tech_name: TechniqueName, model_dir: str, local_load: bool = True) -> None:
        model_name = VUTECO_LM_FND_TECHNIQUES[tech_name]
        model_class, _ = LM_FINDER_MODELS[model_name]
        # TODO Improve management of where to export models
        model_path = os.path.join(model_dir, model_name, FINAL) if local_load else model_dir.format(model=model_name)
        super().__init__(tech_name, model_class.from_pretrained(model_path))


class VutecoGrepFinder(VutecoTechnique):
    def __init__(self, model_dir: str) -> None:
        # TODO Improve management of where to export models
        super().__init__(TechniqueName.GREP, GrepFinder.load_grep_fnd(os.path.join(model_dir, FinderName.GREP_FND)))


class VutecoVocabularyFinder(VutecoTechnique):
    def __init__(self, model_dir: str) -> None:
        # TODO Improve management of where to export models
        super().__init__(TechniqueName.VOCABULARY, VocabularyFinder.load_vocab_fnd(os.path.join(model_dir, FinderName.VOCABULARY_FND)))


class VutecoNeuralNetworkE2E(VutecoTechnique):
    def __init__(self, tech_name: TechniqueName, model_dir: str, local_load: bool = True) -> None:
        model_name = VUTECO_NN_E2E_TECHNIQUES[tech_name]
        model_class, _ = NN_E2E_MODELS[model_name]
        # TODO Improve management of where to export models
        model_path = os.path.join(model_dir, model_name, "e2e") if local_load else model_dir.format(model=model_name)
        super().__init__(tech_name, model_class.from_pretrained(model_path))


class VutecoLanguageModelE2E(VutecoTechnique):
    def __init__(self, tech_name: TechniqueName, model_dir: str, local_load: bool = True) -> None:
        model_name = VUTECO_LM_E2E_TECHNIQUES[tech_name]
        model_class, _ = LM_E2E_MODELS[model_name]
        # TODO Improve management of where to export models
        model_path = os.path.join(model_dir, model_name, "e2e", FINAL) if local_load else model_dir.format(model=model_name)
        super().__init__(tech_name, model_class.from_pretrained(model_path))


class VutecoNeuralNetworkFinderLinker(VutecoTechnique):
    def __init__(self, tech_name: TechniqueName, model_dir: str, local_load: bool = True) -> None:
        model_name = VUTECO_NN_FL_TECHNIQUES[tech_name]
        fnd_model_class, lnk_model_class = NN_FL_MODELS[model_name]
        # TODO Improve management of where to export models
        fnd_model_path = os.path.join(model_dir, model_name.replace("fl", "fnd"), FINAL) if local_load else model_dir.format(model=model_name)
        lnk_model_path = os.path.join(model_dir, model_name.replace("fl", "lnk"), FINAL) if local_load else model_dir.format(model=model_name)
        fnd_model = fnd_model_class.from_pretrained(fnd_model_path)
        lnk_model = lnk_model_class.from_pretrained(lnk_model_path)
        super().__init__(tech_name, FinderLinkerWrapper(fnd_model=fnd_model, lnk_model=lnk_model))


class VutecoLanguageModelFinderLinker(VutecoTechnique):
    def __init__(self, tech_name: TechniqueName, model_dir: str, local_load: bool = True) -> None:
        model_name = VUTECO_LM_FL_TECHNIQUES[tech_name]
        fnd_model_class, lnk_model_class = LM_FL_MODELS[model_name]
        # TODO Improve management of where to export models
        fnd_model_path = os.path.join(model_dir, model_name.replace("fl", "fnd"), FINAL) if local_load else model_dir.format(model=model_name)
        lnk_model_path = os.path.join(model_dir, model_name.replace("fl", "lnk"), FINAL) if local_load else model_dir.format(model=model_name)
        fnd_model = fnd_model_class.from_pretrained(fnd_model_path)
        lnk_model = lnk_model_class.from_pretrained(lnk_model_path)
        super().__init__(tech_name, FinderLinkerWrapper(fnd_model=fnd_model, lnk_model=lnk_model))


class VutecoGrepMatcher(VutecoTechnique):
    def __init__(self, model_dir: str) -> None:
        # TODO Improve management of where to export models
        super().__init__(TechniqueName.GREP, GrepMatcher.load_grep_mtc(os.path.join(model_dir, End2EndName.GREP_E2E)))


class VutecoFixCommit(VutecoTechnique):
    def __init__(self, *args) -> None:
        super().__init__(TechniqueName.FIX, FixCommitModel())

    def __call__(self, tests: list[TestCase], fix_commits: list[git.Commit] = None, *args, **kwargs) -> dict[TestCase, float]:
        if fix_commits is None or len(fix_commits) == 0:
            return {t: 0.0 for t in tests}
        res = {
            t: self.model.get_witnessing_score(fix_commits[0].repo.working_dir, t.file_path, t.method_name, fix_commits)
            for t in tqdm(tests, desc=f"    - Finding witnessing tests using {self.name}")
        }
        self.model.clear_cache()
        return res
