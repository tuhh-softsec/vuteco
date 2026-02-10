import os

from vuteco.core.common.constants import (EVALUATED_MODEL_DIRPATH, End2EndName,
                                          FinderLinkerName, FinderName,
                                          LinkerName, TechniqueName)
from vuteco.core.modeling.modeling_lm_e2e import (CodeLlamaE2EModel,
                                                  DeepSeekCoderE2EModel,
                                                  LanguageModelE2E,
                                                  QwenCoderE2EModel)
from vuteco.core.modeling.modeling_lm_fnd import (CodeLlamaFinder,
                                                  DeepSeekCoderFinder,
                                                  LanguageModelFinder,
                                                  QwenCoderFinder)
from vuteco.core.modeling.modeling_lm_lnk import (CodeLlamaLinker,
                                                  DeepSeekCoderLinker,
                                                  LanguageModelLinker,
                                                  QwenCoderLinker)
from vuteco.core.modeling.modeling_nn_e2e import (CodeBertE2EModel,
                                                  CodeT5PlusE2EModel,
                                                  NeuralNetworkE2E,
                                                  UniXCoderE2EModel)
from vuteco.core.modeling.modeling_nn_fnd import (CodeBertFinder,
                                                  CodeT5PlusFinder,
                                                  NeuralNetworkFinder,
                                                  UniXCoderFinder)
from vuteco.core.modeling.modeling_nn_lnk import (CodeBertLinker,
                                                  CodeT5PlusLinker,
                                                  NeuralNetworkLinker,
                                                  UniXCoderLinker)

CB_FND_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, FinderName.CODEBERT_FND.value)
CT5P_FND_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, FinderName.CODET5PLUS_FND.value)
UXC_FND_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, FinderName.UNIXCODER_FND.value)
CL_FND_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, FinderName.CODELLAMA_FND.value)
QC_FND_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, FinderName.QWENCODER_FND.value)
DSC_FND_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, FinderName.DEEPSEEKCODER_FND.value)
CB_LNK_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, LinkerName.CODEBERT_LNK.value)
CT5P_LNK_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, LinkerName.CODET5PLUS_LNK.value)
UXC_LNK_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, LinkerName.UNIXCODER_LNK.value)
CL_LNK_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, LinkerName.CODELLAMA_LNK.value)
QC_LNK_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, LinkerName.QWENCODER_LNK.value)
DSC_LNK_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, LinkerName.DEEPSEEKCODER_LNK.value)

CB_E2E_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, End2EndName.CODEBERT_E2E.value)
CT5P_E2E_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, End2EndName.CODET5PLUS_E2E.value)
UXC_E2E_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, End2EndName.UNIXCODER_E2E.value)
CL_E2E_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, End2EndName.CODELLAMA_E2E.value)
QC_E2E_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, End2EndName.QWENCODER_E2E.value)
DSC_E2E_EXPORT_EVAL_DIRPATH = os.path.join(EVALUATED_MODEL_DIRPATH, End2EndName.DEEPSEEKCODER_E2E.value)

NN_FINDER_MODELS: dict[FinderName, tuple[NeuralNetworkFinder, str]] = {
    FinderName.CODEBERT_FND: (CodeBertFinder, CB_FND_EXPORT_EVAL_DIRPATH),
    FinderName.CODET5PLUS_FND: (CodeT5PlusFinder, CT5P_FND_EXPORT_EVAL_DIRPATH),
    FinderName.UNIXCODER_FND: (UniXCoderFinder, UXC_FND_EXPORT_EVAL_DIRPATH)
}
LM_FINDER_MODELS: dict[FinderName, tuple[LanguageModelFinder, str]] = {
    FinderName.CODELLAMA_FND: (CodeLlamaFinder, CL_FND_EXPORT_EVAL_DIRPATH),
    FinderName.QWENCODER_FND: (QwenCoderFinder, QC_FND_EXPORT_EVAL_DIRPATH),
    FinderName.DEEPSEEKCODER_FND: (DeepSeekCoderFinder, DSC_FND_EXPORT_EVAL_DIRPATH),
}
NN_LINKER_MODELS: dict[LinkerName, tuple[NeuralNetworkLinker, str]] = {
    LinkerName.CODEBERT_LNK: (CodeBertLinker, CB_LNK_EXPORT_EVAL_DIRPATH),
    LinkerName.CODET5PLUS_LNK: (CodeT5PlusLinker, CT5P_LNK_EXPORT_EVAL_DIRPATH),
    LinkerName.UNIXCODER_LNK: (UniXCoderLinker, UXC_LNK_EXPORT_EVAL_DIRPATH)
}
LM_LINKER_MODELS: dict[LinkerName, tuple[LanguageModelLinker, str]] = {
    LinkerName.CODELLAMA_LNK: (CodeLlamaLinker, CL_LNK_EXPORT_EVAL_DIRPATH),
    LinkerName.QWENCODER_LNK: (QwenCoderLinker, QC_LNK_EXPORT_EVAL_DIRPATH),
    LinkerName.DEEPSEEKCODER_LNK: (DeepSeekCoderLinker, DSC_LNK_EXPORT_EVAL_DIRPATH),
}
NN_E2E_MODELS: dict[End2EndName, tuple[NeuralNetworkE2E, str]] = {
    End2EndName.CODEBERT_E2E: (CodeBertE2EModel, CB_E2E_EXPORT_EVAL_DIRPATH),
    End2EndName.CODET5PLUS_E2E: (CodeT5PlusE2EModel, CT5P_E2E_EXPORT_EVAL_DIRPATH),
    End2EndName.UNIXCODER_E2E: (UniXCoderE2EModel, UXC_E2E_EXPORT_EVAL_DIRPATH)
}
LM_E2E_MODELS: dict[End2EndName, tuple[LanguageModelE2E, str]] = {
    End2EndName.CODELLAMA_E2E: (CodeLlamaE2EModel, CL_E2E_EXPORT_EVAL_DIRPATH),
    End2EndName.QWENCODER_E2E: (QwenCoderE2EModel, QC_E2E_EXPORT_EVAL_DIRPATH),
    End2EndName.DEEPSEEKCODER_E2E: (DeepSeekCoderE2EModel, DSC_E2E_EXPORT_EVAL_DIRPATH)
}
NN_FL_MODELS: dict[FinderLinkerName, tuple[NeuralNetworkFinder, NeuralNetworkLinker]] = {
    FinderLinkerName.CODEBERT_FL: (CodeBertFinder, CodeBertLinker),
    FinderLinkerName.CODET5PLUS_FL: (CodeT5PlusFinder, CodeT5PlusLinker),
    FinderLinkerName.UNIXCODER_FL: (UniXCoderFinder, UniXCoderLinker),
}
LM_FL_MODELS: dict[FinderLinkerName, tuple[LanguageModelFinder, LanguageModelLinker]] = {
    FinderLinkerName.CODELLAMA_FL: (CodeLlamaFinder, CodeLlamaLinker),
    FinderLinkerName.QWENCODER_FL: (QwenCoderFinder, QwenCoderLinker),
    FinderLinkerName.DEEPSEEKCODER_FL: (DeepSeekCoderFinder, DeepSeekCoderLinker),
}

VUTECO_NN_FND_TECHNIQUES: dict[TechniqueName, FinderName] = {
    TechniqueName.CODEBERT: FinderName.CODEBERT_FND,
    TechniqueName.CODET5PLUS: FinderName.CODET5PLUS_FND,
    TechniqueName.UNIXCODER: FinderName.UNIXCODER_FND
}
VUTECO_LM_FND_TECHNIQUES: dict[TechniqueName, FinderName] = {
    TechniqueName.CODELLAMA: FinderName.CODELLAMA_FND,
    TechniqueName.QWENCODER: FinderName.QWENCODER_FND,
    TechniqueName.DEEPSEEKCODER: FinderName.DEEPSEEKCODER_FND
}
VUTECO_NN_E2E_TECHNIQUES: dict[TechniqueName, End2EndName] = {
    TechniqueName.CODEBERT: End2EndName.CODEBERT_E2E,
    TechniqueName.CODET5PLUS: End2EndName.CODET5PLUS_E2E,
    TechniqueName.UNIXCODER: End2EndName.UNIXCODER_E2E
}
VUTECO_LM_E2E_TECHNIQUES: dict[TechniqueName, End2EndName] = {
    TechniqueName.CODELLAMA: End2EndName.CODELLAMA_E2E,
    TechniqueName.QWENCODER: End2EndName.QWENCODER_E2E,
    TechniqueName.DEEPSEEKCODER: End2EndName.DEEPSEEKCODER_E2E
}
VUTECO_NN_FL_TECHNIQUES: dict[TechniqueName, FinderLinkerName] = {
    TechniqueName.CODEBERT: FinderLinkerName.CODEBERT_FL,
    TechniqueName.CODET5PLUS: FinderLinkerName.CODET5PLUS_FL,
    TechniqueName.UNIXCODER: FinderLinkerName.UNIXCODER_FL
}
VUTECO_LM_FL_TECHNIQUES: dict[TechniqueName, FinderLinkerName] = {
    TechniqueName.CODELLAMA: FinderLinkerName.CODELLAMA_FL,
    TechniqueName.QWENCODER: FinderLinkerName.QWENCODER_FL,
    TechniqueName.DEEPSEEKCODER: FinderLinkerName.DEEPSEEKCODER_FL
}
