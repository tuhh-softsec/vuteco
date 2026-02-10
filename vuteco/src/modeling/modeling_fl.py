from typing import Union

from modeling.modeling_lm_fnd import LanguageModelFinder
from modeling.modeling_lm_lnk import LanguageModelLinker
from modeling.modeling_nn_fnd import NeuralNetworkFinder
from modeling.modeling_nn_lnk import NeuralNetworkLinker


class FinderLinkerWrapper():

    def __init__(self,
                 fnd_model: Union[NeuralNetworkFinder, LanguageModelFinder],
                 lnk_model: Union[NeuralNetworkLinker, LanguageModelLinker]):
        self.fnd_model = fnd_model
        self.lnk_model = lnk_model

    def get_witnessing_score(self, test_code: Union[str, list[str]]) -> Union[float, list[float]]:
        return self.fnd_model.get_witnessing_score(test_code)

    def get_relation_score(self, test_code: Union[str, list[str]], vuln: str) -> Union[float, list[float]]:
        # NOTE Not really batched, but it does not matter now, it's just a matter of performance
        if isinstance(test_code, list):
            wit_scores = self.get_witnessing_score(test_code)
            return [self.lnk_model.get_relation_score(tc, vuln) if s >= 0.5 else 0.0 for s, tc in zip(wit_scores, test_code)]
        else:
            if self.get_witnessing_score(test_code) >= 0.5:
                return self.lnk_model.get_relation_score(test_code, vuln)
            return 0.0
