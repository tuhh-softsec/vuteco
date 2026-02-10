import json
import os
import tempfile
from typing import Optional

import pandas as pd
from git import Commit, Repo
from halo import Halo
from vuteco.core.common.constants import (DEFAULT_VUTECO_RESULT_DIRNAME,
                                          TechniqueName, VutecoRevisionStyle)
from vuteco.core.common.utils_mining import (collect_tests_in_revision,
                                             get_cve_descr_date, get_cwe_info,
                                             get_cwe_name, join_cwe_id_info)
from vuteco.main.vuteco_domain import TestCase, VutecoTechnique


class CommitEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Commit):
            return obj.hexsha
        return super(CommitEncoder, self).default(obj)


def prepare_vulns_to_match(repo: Repo, input_entry: dict, knowledge_base: dict, include_cwe: bool = False) -> dict[str, dict]:
    if "vuln_ids" not in input_entry:
        print(f"Without info about the vulnerability (either \"vuln_ids\" or \"descriptions\") VUTECO cannot link tests to vulnerability as requested. Skipping.")
        return {}
    vulnerabilities = {}
    for vuln_id in input_entry["vuln_ids"].split(","):
        vulnerabilities[vuln_id] = {}
        if vuln_id in knowledge_base:
            vulnerabilities[vuln_id]["description"] = knowledge_base[vuln_id]["description"]
            fixes = []
            for f in knowledge_base[vuln_id]["fixes"]:
                try:
                    fixes.append(repo.commit(f["hash"]))
                except:
                    pass
            vulnerabilities[vuln_id]["fixes"] = fixes
            vulnerabilities[vuln_id]["cwes"] = ""
            if include_cwe:
                if len(knowledge_base[vuln_id]["cwe_id"]) > 0:
                    vulnerabilities[vuln_id]["cwes"] = ",".join(join_cwe_id_info(c, n) for c, n in zip(knowledge_base[vuln_id]["cwe_id"], knowledge_base[vuln_id]["cwe_name"]))
        else:
            print(f"Vulnerability {vuln_id} not in the knowledge base. Searching data from CVE Search...")
            try:
                descr, _ = get_cve_descr_date(vuln_id)
                vulnerabilities[vuln_id]["description"] = descr if descr else ""
                vulnerabilities[vuln_id]["cwes"] = ""
                if include_cwe:
                    cwes = get_cwe_info(vuln_id)
                    if len(cwes) > 0:
                        vulnerabilities[vuln_id]["cwes"] = ",".join(join_cwe_id_info(ci["cweId"], get_cwe_name(ci["cweId"])) for ci in cwes)
            except:
                print(f"Failed to fetch info about {vuln_id} from CVE Search. Skipping.")
                del vulnerabilities[vuln_id]
    return vulnerabilities


def add_scores(test_results: dict[TestCase, dict], tech_name: TechniqueName, tech_output: dict[TestCase, float], field_name: str, v_id: Optional[str] = None):
    scores = []
    for tc, score in tech_output.items():
        if tech_name not in test_results[tc][field_name]:
            test_results[tc][field_name][tech_name] = {}
        if v_id:
            test_results[tc][field_name][tech_name][v_id] = round(score, 3)
        else:
            test_results[tc][field_name][tech_name] = round(score, 3)
        scores.append(round(score, 3))
    nr_tests_flagged = sum(1 for sc in scores if sc > 0.01)
    print(f"      - Flagged {nr_tests_flagged} test(s) with at least 0.01 score.")


def inspect_revision(reponame: str,
                     revision: Commit,
                     techniques: list[VutecoTechnique],
                     vuln_finding: bool,
                     vuln_matching: bool,
                     batched_inference: bool,
                     vulnerabilities: dict[str, dict]):
    tests_in_revision = collect_tests_in_revision(revision)
    test_cases = [
        TestCase(reponame, t["file"], t["class"], t["method"], t["code"], t["startline"])
        for t in tests_in_revision
        if t["code"]
    ]
    revision_output = {
        "stats": {"nrTests": len(test_cases)},
        "judgments": {}
    }
    if len(test_cases) == 0:
        print(f"  - Found no tests. Cannot make any analyses in this revision without tests.")
        return revision_output
    print(f"  - Found {len(test_cases)} tests to inspect.")
    test_results: dict[TestCase, dict] = {
        tc: {
            **vars(tc),
            **({"witnessing_scores": {}} if vuln_finding else {}),
            **({"matched_vulns": {}} if vuln_matching else {})
        }
        for tc in test_cases
    }
    for tech in techniques:
        print(f"  - Inspecting tests using \"{tech.name}\"")
        if vuln_finding:
            print(f"    - Finding witnessing tests")
            tech_output = tech(tests=test_cases, batched_inference=batched_inference)
            add_scores(test_results, tech.name, tech_output, "witnessing_scores")
        if vuln_matching:
            for v_id, v_info in vulnerabilities.items():
                print(f"    - Matching tests to {v_id}")
                tech_output = tech(
                    tests=test_cases,
                    description=v_info["description"],
                    fix_commits=v_info["fixes"],
                    cwes=v_info["cwes"],
                    batched_inference=batched_inference
                )
                add_scores(test_results, tech.name, tech_output, "matched_vulns", v_id)
    revision_output["judgments"] = {tc.id: data for tc, data in test_results.items()}
    return revision_output


def vuteco_start(input_df: pd.DataFrame, output_dirpath: str, techniques: list[VutecoTechnique], vulnerability_kb: dict, revision_style: VutecoRevisionStyle, vuln_matching: bool = True, vuln_finding: bool = True, include_cwe: bool = False, batched_inference: bool = False, skip_inspected_projects: bool = True):
    if output_dirpath:
        print(f"The result of each project analyzed will be stored as JSON files in directory '{output_dirpath}'")
    else:
        output_dirpath = os.path.join(os.getcwd(), DEFAULT_VUTECO_RESULT_DIRNAME)
        print(f"The results of each project will be stored as JSON in the current working directory under '{DEFAULT_VUTECO_RESULT_DIRNAME}'")
    os.makedirs(output_dirpath, exist_ok=True)
    for idx, input_entry in enumerate(input_df.to_dict(orient="records")):
        str_prefix = f"({idx + 1}/{len(input_df)})"
        clone_url = input_entry["url"]
        print("")
        if clone_url is None:
            print(f"{str_prefix} The field \"url\" is missing in the input file. Skipping.")
            continue
        reponame = os.path.sep.join(clone_url.split(os.path.sep)[-2:])
        org_name, project_name = os.path.split(reponame)
        org_output_dirpath = os.path.join(output_dirpath, org_name)
        project_output_filepath = os.path.join(org_output_dirpath, f"{project_name}.json")
        if skip_inspected_projects and os.path.exists(project_output_filepath):
            print(f"{str_prefix} Project \"{reponame}\" already inspected. Skipping as requested.")
            continue

        project_results = {}
        if os.path.exists(project_output_filepath):
            with open(project_output_filepath, "r") as fin:
                try:
                    project_results = json.load(fin)
                except:
                    project_results = {
                        "repo": reponame,
                        "remote": clone_url
                    }
        else:
            project_results = {
                "repo": reponame,
                "remote": clone_url
            }
        with tempfile.TemporaryDirectory() as tmp_dirname:
            with Halo(text=f"{str_prefix} Cloning project {reponame} from {clone_url}") as spinner:
                try:
                    repo = Repo.clone_from(clone_url, tmp_dirname)
                except:
                    spinner.fail("Clone failed :( Going to the next entry.")
                    continue
                spinner.succeed(f"{str_prefix} Project {reponame} cloned!")

            if vuln_matching:
                print("Requested matching tests with vulnerabilities.")
                vulnerabilities = prepare_vulns_to_match(repo, input_entry, vulnerability_kb, include_cwe)
                if len(vulnerabilities) == 0:
                    print(f"Could not retrieve any vulnerability data for analysis. Skipping.")
                    continue
                project_results["candidate_vulns"] = vulnerabilities
            else:
                vulnerabilities = {}

            if revision_style == VutecoRevisionStyle.ALL:
                main_revisions = list(repo.iter_commits())
            elif revision_style == VutecoRevisionStyle.HEAD:
                try:
                    main_revisions = [repo.head.commit]
                except ValueError:
                    print(f"Could not find HEAD revision as requested.")
            elif revision_style == VutecoRevisionStyle.INPUT_FILE:
                if "revision" not in input_entry or input_entry["revision"] is None:
                    print(f"Missing or empty field \"revision\" in input file.")
                try:
                    main_revisions = [repo.commit(input_entry["revision"])]
                except:
                    print(f"Could not find the request revision \"{input_entry['revision']}\".")

            # NOTE There might be projects not related to Java. It's okay, they will be "ignored" by VUTECO at runtime (that will not find any test in those repositories).
            project_results["revisions"] = {}
            print(f"Inspecting {len(main_revisions)} revision(s):")
            for revision in main_revisions:
                print(f"- Inspecting revision {revision.hexsha[:6]}.")
                project_results["revisions"][revision.hexsha] = inspect_revision(reponame, revision, techniques, vuln_finding, vuln_matching, batched_inference, vulnerabilities)
                os.makedirs(org_output_dirpath, exist_ok=True)
                with open(project_output_filepath, "w") as fout:
                    json.dump(project_results, fout, indent=2, cls=CommitEncoder)

            if vuln_matching:
                project_results["revisions_after_patch"] = {}
                print(f"Inspecting {len(project_results['candidate_vulns'])} revision(s) after vulnerability patches:")
                for v_id, v_info in vulnerabilities.items():
                    if len(v_info["fixes"]) == 0:
                        continue
                    last_fix = max(v_info["fixes"], key=lambda x: x.committed_datetime)
                    print(f"- Inspecting revision {last_fix.hexsha[:6]} for {v_id}.")
                    project_results["revisions_after_patch"][v_id] = {
                        "last_patch": last_fix.hexsha,
                        **inspect_revision(reponame, last_fix, techniques, vuln_finding, vuln_matching, batched_inference, {v_id: v_info})
                    }
                    os.makedirs(org_output_dirpath, exist_ok=True)
                    with open(project_output_filepath, "w") as fout:
                        json.dump(project_results, fout, indent=2, cls=CommitEncoder)
