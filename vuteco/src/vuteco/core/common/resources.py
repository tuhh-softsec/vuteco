import json
from importlib import resources


def get_vuln_kb() -> dict:
    with resources.files(f"{__package__}.res").joinpath("vutecokb.json").open("r") as f:
        return json.load(f)


def get_vuln_kb_path() -> str:
    return resources.files(f"{__package__}.res").joinpath("vutecokb.json")


def get_substring_ext_patterns() -> str:
    with resources.files(f"{__package__}.res.patterns").joinpath("substring-match-ext").open("r") as f:
        return f.read()


def get_substring_red_patterns() -> str:
    with resources.files(f"{__package__}.res.patterns").joinpath("substring-match-red").open("r") as f:
        return f.read()


def get_exact_patterns() -> str:
    with resources.files(f"{__package__}.res.patterns").joinpath("exact-match").open("r") as f:
        return f.read()
