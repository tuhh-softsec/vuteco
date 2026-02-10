import json
import os
import re

import yaml
from core.common.constants import (PROJECT_KB_FILEPATH,
                                   PROJECT_KB_REPO_DIRPATH, PROJECT_KB_URL,
                                   PROJECTKB_BRANCH, REEF_KB_FILEPATH,
                                   REEF_RAW_FILEPATH, REPOSVUL_KB_FILEPATH,
                                   REPOSVUL_RAW_FILEPATH, VUTECO_KB_FILEPATH)
from core.common.utils_mining import (get_cve_descr_date, get_cwe_info,
                                      get_cwe_name, normalize_repo_url)
from git import Repo
from halo import Halo
from tqdm import tqdm


def get_new_fixes(existing: list[dict], new: list[dict]):
    new_fixes = []
    for new_f in new:
        matched = False
        for host_f in existing:
            if normalize_repo_url(host_f["url"]) == normalize_repo_url(new_f["url"]) and host_f["hash"][:8] == new_f["hash"][:8]:
                matched = True
                break
        if not matched:
            new_fixes.append(new_f)
    return new_fixes


def build_projectkb(dest_dir: str):
    if not os.path.exists(PROJECT_KB_REPO_DIRPATH):
        try:
            with Halo(text=f"Cloning ProjectKB repository ({PROJECT_KB_URL})") as spinner:
                repo = Repo.clone_from(PROJECT_KB_URL, PROJECT_KB_REPO_DIRPATH)
                spinner.succeed("ProjectKB cloned!")
        except:
            print(f"Failed to clone from {PROJECT_KB_URL}. Cannot extract vulnerabilities from ProjectKB.")
            return
    else:
        repo = Repo(PROJECT_KB_REPO_DIRPATH)
    repo.git.checkout(PROJECTKB_BRANCH, "--")
    statements: list[dict] = []
    yaml_files = []
    for root, _, files in os.walk(PROJECT_KB_REPO_DIRPATH):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                yaml_files.append(os.path.join(root, file))
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as fin:
            statements.append(yaml.safe_load(fin))
    project_kb = {}
    for st in tqdm(statements, desc="Reading vulnerability statements"):
        cve_id = st.get("vulnerability_id")
        if cve_id:
            fixes = []
            for f in st.get("fixes", []):
                for c in f.get("commits", []):
                    fixes.append({
                        "url": normalize_repo_url(c.get("repository")),
                        "hash": c.get("id")
                    })
            notes = st.get("notes", [])
            api_descr, pub_date = get_cve_descr_date(cve_id)
            descr = notes[0].get("text") if len(notes) > 0 else api_descr
            cwes = []
            cwe_names = []
            cwe_info = get_cwe_info(cve_id)
            for ci in cwe_info:
                cwes.append(ci["cweId"])
                cwe_names.append(get_cwe_name(ci["cweId"]))
            project_kb[cve_id] = {
                "description": descr,
                "publication_date": pub_date,
                "cwe_id": cwes,
                "cwe_name": cwe_names,
                "fixes": fixes
            }
    with open(dest_dir, "w") as fout:
        json.dump(project_kb, fout, indent=2)


def build_standard(dest_dir: str, in_file: str, cve_key: str, cve_descr_key: str, cwe_key: str, fix_key: str):
    with open(in_file, 'r', encoding='utf-8') as f:
        raw_dataset: list[dict] = [json.loads(line) for line in f]
    clean_kb = {}
    for a_line in tqdm(raw_dataset, desc="Processing vulnerability entries"):
        cve_id = a_line[cve_key]
        descr = None
        if cve_descr_key in a_line:
            descr = a_line[cve_descr_key]
        api_descr, pub_date = get_cve_descr_date(cve_id)
        if not descr:
            descr = api_descr
        cwes = []
        cwe_names = []
        for cwe in a_line[cwe_key]:
            cwes.append(cwe)
            cwe_names.append(get_cwe_name(cwe))
        fix_url: str = a_line[fix_key]
        url, hash = tuple(fix_url.rsplit("/commit/"))
        fixes = [{
            "url": normalize_repo_url(url),
            "hash": hash
        }]
        if not cve_id in clean_kb:
            clean_kb[cve_id] = {
                "description": descr,
                "publication_date": pub_date,
                "cwe_id": cwes,
                "cwe_name": cwe_names,
                "fixes": fixes
            }
        else:
            clean_kb[cve_id]["fixes"].extend(get_new_fixes(clean_kb[cve_id]["fixes"], fixes))
    with open(dest_dir, "w") as fout:
        json.dump(clean_kb, fout, indent=2)


def build_reefkb(dest_dir: str):
    build_standard(dest_dir, in_file=REEF_RAW_FILEPATH, cve_key="cve_id", cve_descr_key=None, cwe_key="CWEs", fix_key="html_url")


def build_reposvulkb(dest_dir: str):
    build_standard(dest_dir, in_file=REPOSVUL_RAW_FILEPATH, cve_key="cve_id", cve_descr_key="cve_description", cwe_key="cwe_id", fix_key="html_url")


def merge_kbs(existing: dict, new: dict):
    merged = existing.copy()
    for cve_id, cve_data in new.items():
        if cve_id not in merged:
            merged[cve_id] = cve_data
            continue
        elif merged[cve_id] == cve_data:
            continue
        if not merged[cve_id]["description"]:
            merged[cve_id]["description"] = cve_data["description"]
        for cwe_id, cwe_name in zip(cve_data["cwe_id"], cve_data["cwe_name"]):
            if cwe_id not in merged[cve_id]["cwe_id"]:
                merged[cve_id]["cwe_id"].append(cwe_id)
                merged[cve_id]["cwe_name"].append(cwe_name)
        merged[cve_id]["fixes"].extend(get_new_fixes(merged[cve_id]["fixes"], cve_data["fixes"]))
    return merged


if __name__ == "__main__":
    if not os.path.exists(PROJECT_KB_FILEPATH):
        build_projectkb(PROJECT_KB_FILEPATH)
    if not os.path.exists(REEF_KB_FILEPATH):
        build_reefkb(REEF_KB_FILEPATH)
    if not os.path.exists(REPOSVUL_KB_FILEPATH):
        build_reposvulkb(REPOSVUL_KB_FILEPATH)

    with open(PROJECT_KB_FILEPATH) as fin:
        project_kb: dict = json.load(fin)
    with open(REEF_KB_FILEPATH) as fin:
        reef_kb: dict = json.load(fin)
    with open(REPOSVUL_KB_FILEPATH) as fin:
        reposvul_kb: dict = json.load(fin)
    # TODO Add CVEFixes? It's an SQL DB to query... maybe later
    vuteco_kb = merge_kbs(project_kb, reef_kb)
    vuteco_kb = merge_kbs(vuteco_kb, reposvul_kb)
    with open(VUTECO_KB_FILEPATH, "w") as fout:
        json.dump(vuteco_kb, fout, indent=2)
