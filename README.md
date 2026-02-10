# VuTeCo - AI-driven Collector of Vulnerability-witnessing Tests

This repository contains the maintained source code of **VuTeCo** (VUlnerability TEst COllector), which can scan Java Git repositories to (1) find security-related test cases and (2) match them with CVE identifiers.

Please, see the [MSR'26 paper](https://arxiv.org/abs/2502.03365) for more details about its inner workings.

If you are looking for the MSR'26 version of VuTeCo, please see the [Zenodo package](https://doi.org/10.5281/zenodo.18258566).

If you are looking for the dataset **Test4Vul**, containing some manually-validated tests found in the wild by VuTeCo, please see https://github.com/tuhh-softsec/test4vul.

# Table of Contents
1. [Requirements](#requirements)
2. [Installation](#install)
3. [Running VuTeCo](#running)
4. [Supported Models](#models)
5. [How to Extend VuTeCo with a new Model/Technique](#extend)
6. [Future Work](#future)

# Requirements

<a name="requirements"></a>

These are the base requirements to run VuTeCo:

- Python 3.10.0
- Java 8+
- A **stable Internet connection** (e.g., for downloading the Python packages and cloning remote repositories during inference).

VuTeCo has been tested on a Linux-based OS so far. Nevertheless, the scripts were implemented to be OS-agnostic, so they should also work on MacOS or Windows.

**NOTES**:
- The following commands assumes that `python` is the default alias for the selected Python installation. You can change to `python3` without issues.
- The `XX` indicate the [acronym](#acronyms) of an AI model supported in VuTeCo.
- The description of all command-line arguments can be found in the file `vuteco/common/cli_args.py`.

# Installation

<a name="install"></a>

Ensure to have sufficient space to host the packages downloaded from PyPI (roughly 7 GB).
- This setup will be improved in the future to avoid installing unneeded dependencies if one wants to use VuTeCo without the training-evaluation pipeline.

Move into the `vuteco/` directory:

```sh
cd vuteco/
```

If this is the first use of this package, create the virtual environment and activate it.

```sh
python -m venv ./venv
source venv/bin/activate
```

Ensure the right version of `setuptools` and and up to date version of `wheel`. Note that a too recent version of `setuptools` (`>=82.0.0`) may have problem with some dependencies still using `pkg_resources` (they will be replaced in the future):

```sh
python -m pip install --upgrade pip "setuptools<81.0.0" wheel
```

Install the required dependencies in the virtual environment (can take some minutes), as listed in `pyproject.toml`:

```sh
python -m pip install -e .
```

After this, VuTeCo can be run with the command `vuteco`, which is equivalent to `python -m vuteco.cli` (you can choose any).

## Troubleshooting

If you get unexpected problems caused by dependencies, use the pinned versions listed in `requirements.txt`

```sh
python -m pip install -r requirements.txt
```

If `requirements.txt` happens to have the line `pkg_resources==0.0.0`, remove it.

If packages like `halo` or `ares` are raising issues because of `wheel`, run this and try again:

```sh
python -m pip install --upgrade pip "setuptools<81.0.0" wheel
```

Unsloth could give problems with dependencies. Try to ensure always the latest version directly from the repository if the one in the `requirements.txt` is giving issues:

```sh
python -m pip uninstall unsloth unsloth-zoo -y && python -m pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git git+https://github.com/unslothai/unsloth-zoo.gi
```

# Running VuTeCo

<a name="running"></a>

The command to run VuTeCo is `vuteco`, which points to `vuteco/vuteco/cli.py`. The arguments it accepts are described in file `vuteco/common/cli_args.py`.

VuTeCo can be run in two modes:
- **Finding**: it predicts whether all the JUnit test methods found in a given project repository are security-related (returns a probability).
- **Matching**: it predicts whether all the JUnit test methods found in a given project repository and the supplied list of CVEs are matched (returns a probability).

The user can interpret the predicted probabilities freely (the tool per se does not classify, but only return probabilities). The recommended threshold for the positive classifications is the default 0.5.

The input projects to analyze can be supplied through the command-line argument `-i` (see [this example](`examples/input.csv`) as a guidance). The output is returned as JSON files, one per project analyzed, and placed in the directory indicated by the command-line argument `-o`.
- Make sure the directory does not exist or is empty so it does not clash with your existing files.

The AI model to use can be set with the argument `-t`. The list of supported models is reported [below](#models).
- In Finding mode, the recommended model is `uxc` (UniXcoder);
- In Matching mode, the recommended model is `dsc` (DeepSeek Coder).

## Sample Usages

VuTeCo can be called in Finding mode in this way (with model `uxc`):

```sh
vuteco -i <PROJECT-LIST-FILE> -o <OUTPUT-DIRECTORY> -r from-file --no-vuln-match --skip-inspected-projects -t uxc
```

VuTeCo can be called in Matching mode in this way (with model `dsc`):

```sh
vuteco -i <PROJECT-LIST-FILE> -o <OUTPUT-DIRECTORY> -r from-file --no-vuln-find --skip-inspected-projects -t dsc
```

## Troubleshooting

VuTeCo automatically downloads the weights for the Finding and Matching models from Hugging Face (https://huggingface.co/emaiannone/models). However, should this automatic download encounter issues, you can solve this by:
1. Manually download the model from the Hugging Face web pages.
2. Place the Finding models under `<ANY-DIR>/<MODEL-NAME>/final`; place the Matching models under `<ANY-DIR>/<MODEL-NAME>/e2e/final` (this weird path naming will be solved in the future).
3. Add this argument to the above commands: `--model-dir <ANY-DIR>`.

# Supported Models

<a name="models"></a>

| Acronym | Name |
|---|---|
| cb | CodeBERT |
| uxc | UniXcoder |
| ct5p | CodeT5+ |
| cl | CodeLlama |
| dsc | DeepSeek Coder |
| qc | Qwen Coder |

# How to Extend VuTeCo with a new Model/Technique

<a name="extend"></a>

This guide explains how to add a new model or technique to **VuTeCo**. This guide will be further improved in the future.

Assume you want to add a technique named `MyPowerfulTechnique`, with the acronym `mpt`.

## 1. Update `constants.py`

1. **Register the technique for training and evaluation**
   Add the name `mpt` to:
   - `FinderName` if it is for the *Finding* task, or
   - `End2EndName` if it is for the *Matching* task.
   
   Follow the naming conventions used by existing entries (recommended, but not mandatory), e.g., `MYPOWERFULTECHNIQUE_FND = "mpt-fnd"` or `MYPOWERFULTECHNIQUE_E2E = "mpt-e2e"`.

2. **Register the technique for inference**
   Add the technique name to `TechniqueName`. Again, following the existing naming patterns where possible, e.g., `MYPOWERFULTECHNIQUE = "mpt"`

## 2. Implement the core of your technique

Create the class that implements your technique under the `modeling` module, following the style of the existing files.

This is where you define the specific behavior of your technique. If you want it to behave like the existing models in VuTeCo, ensure to **inherit from the existing superclasses** (e.g., `NeuralNetworkFinder`, `LanguageModelFinder`, `NeuralNetworkE2E`, `LanguageModelE2E`). If so, place the class in the appropriate file. For example, an LLM for Matching should typically go in `modeling_lm_e2e.py`. Follow existing naming conventions where possible. Otherwise, you have to define the custom behavior on your own, possibly in a new file if desired.

## 3. Update `cli_constants.py`

1. **Define an evaluation output directory**
   Create a new constant pointing to the directory where evaluation results will be written.
   Follow the naming convention used by existing constants, e.g., `MPT_FND_EXPORT_EVAL_DIRPAT = os.path.join(EVALUATED_MODEL_DIRPATH, FinderName.MYPOWERFULTECHNIQUE_FND.value)`.

2. **Map the approach to its class and evaluation directory**
   Add an entry to the appropriate `XYZ_MODELS` dictionary, depending on the technique type. For example:
   - `NN_FINDER_MODELS` for neural-network–based Finding approaches
   - `LM_E2E_MODELS` for LLM-based Matching approaches

3. **Map the approach to its inference name**
   Add an entry to the appropriate `VUTECO_XYZ` dictionary to associate the training/evaluation name with the inference name.

# Future Work

This repository is under improvement. These are some activities that will be done to improve the usability of VuTeCo and the clarity of this README:

- Adjust the names with the ones used in the MSR'26 paper (e.g., End2End becomes Matcher)
- Explain better how VuTeCo can be extended (possibly simplifying the code as well).
- Separate VuTeCo (the tool used for inference) from the training-evaluation pipeline (for exporting the models). This also includes separating the required dependencies.
- Clean up dependencies and provide the Docker images to run VuTeCo out of the box.
- Handle *testng* test cases, other than JUnit.

Please, open new issues for suggestions and bug fixes! This is very appreciated :)
