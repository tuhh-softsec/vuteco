# VuTeCo

This repository contains the maintained source code of **VuTeCo** (VUlnerability TEst COllector), which can scan Java Git repositories to (1) find security-related test cases and (2) match them with CVE identifiers.

Please, see the [MSR'26 paper](https://arxiv.org/abs/2502.03365) for more details about its inner workings.

If you are looking for the MSR'26 version of VuTeCo, please see the [Zenodo package](https://doi.org/10.5281/zenodo.18258566).

If you are looking for the dataset **Test4Vul**, containing some manually-validated tests found in the wild by VuTeCo, please see https://github.com/tuhh-softsec/test4vul.

# Requirements

To run VuTeCo, you need the following requirements:

- Python 3.10.0
- Java 8+
- A **stable Internet connection** (e.g., for downloading the Python packages and cloning remote repositories during inference).

VuTeCo has been used on a Linux-based OS so far. Nevertheless, the scripts were implemented to be OS-agnostic, so they should also work on MacOS or Windows.

**NOTES**:
- The following commands assume that `python` is the default alias for the selected Python installation. It can be changed to `python3` without issues.
- The `XX` indicates the acronym of an AI model. See [below](#models) for details.
- The meaning of all command-line arguments can be found in the file `scripts/common/cli_args.py`.

# Set Up

Move into the `scripts/` directory:

```sh
cd scripts/
```

If this is the first use of VuTeCo, create the virtual environment, activate it, and install all the required dependencies in the virtual environment (can take some minutes):

```sh
python -m venv ./venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Troubleshooting

If packages like `halo` or `ares` are complaining because of `wheel`, run this and try again:

```sh
pip install --upgrade pip setuptools wheel
```

If `requirements.txt` has this line `pkg_resources==0.0.0`, remove it.

Unsloth could give problems with dependencies. Try to ensure always the latest version directly from the repository if the one in the `requirements.txt` is giving issues:

```sh
python -m pip uninstall unsloth unsloth-zoo -y && python -m pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git git+https://github.com/unslothai/unsloth-zoo.gi
```

Ensure to have additional space to host the packages downloaded from PyPI (roughly 7 GB).
- This setup will be improved in the future to avoid installing unneeded dependencies if one wants to use VuTeCo without the training-evaluation pipeline.

# Running VuTeCo

The main script for running VuTeCo is `vuteco.py`. It accepts the following arguments (see the complete description in [this file](https://github.com/tuhh-softsec/vuteco/blob/main/scripts/cli_args.py#400)).

VuTeCo can be run in two modes:
- **Finding**: it predicts whether all the JUnit test methods found in a given project repository are security-related (returns a probability).
- **Matching**: it predicts whether all the JUnit test methods found in a given project repository and the supplied list of CVEs are matched (returns a probability).

The user can interpret the predicted probabilities freely (the tool per se does not classify, but only return probabilities). The recommended threshold for the positive classifications is the default 0.5.

The input projects to analyze can be supplied through the command-line argument `-i` (see [this example](`examples/input.csv`) as guidance). The output is returned as JSON files, one per project analyzed, and put in the directory indicated by the command-line argument `-o`. Make sure the directory does not exist or is empty so it does not clash with your existing files.

The AI model to use can be set with the argument `-t`. The list of supported models is reported [below](#models).
- In Finding mode, the recommended model is `uxc` (UniXcoder);
- In Matching mode, the recommended model is `dsc` (Deep Seek Coder).

## Sample Usages

VuTeCo can be called in Finding mode in this way (with model `uxc`):

```sh
python vuteco.py -i <PROJECT-LIST> -o <OUTPUT-DIRECTORY> -r from-file --no-vuln-match --skip-inspected-projects -t uxc
```

VuTeCo can be called in Matching mode in this way (with model `dsc`):

```sh
python vuteco.py -i <PROJECT-LIST> -o <OUTPUT-DIRECTORY> -r from-file --no-vuln-find --skip-inspected-projects -t dsc
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

# Future Work

This repository is under improvement. These are some activities that will be done to improve the usability of VuTeCo and the clarity of this README:

- Separate VuTeCo (the tool used for inference) from the training-evaluation pipeline (for exporting the models).
- Provide the Docker images
- Handle *testng* test cases, other than JUnit.

Please, open new issues for suggestions and bug fixes! This is very appreciated :)
