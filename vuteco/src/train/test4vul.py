import json
import os

import pandas as pd
from core.common.constants import (INTHEWILD_FINDING_DIRPATH,
                                   INTHEWILD_FINDINGS_INSPECTED_FILENAME,
                                   INTHEWILD_MATCHING_DIRPATH,
                                   INTHEWILD_MATCHINGS_INSPECTED_FILENAME,
                                   INTHEWILD_RESULTS_DIRNAME,
                                   TEST4VUL_FILEPATH)


def extract_tests(raw_results_dirpath, confirmed_df: pd.DataFrame, existing_retrieved_tests: dict = {}):
    all_repos = confirmed_df["repo"].tolist()
    retrieved_tests = existing_retrieved_tests.copy()
    for root, _, files in os.walk(raw_results_dirpath):
        for f in files:
            with open(os.path.join(root, f)) as fin:
                project_results = json.load(fin)
                if project_results["repo"] in all_repos:
                    hit_cases = confirmed_df[confirmed_df["repo"] == project_results["repo"]].to_dict(orient="records")
                    for hit_case in hit_cases:
                        if hit_case["revision"] in project_results["revisions"]:
                            project_judgments = project_results["revisions"][hit_case["revision"]]["judgments"]
                            for k, test in project_judgments.items():
                                if test["file_path"] == hit_case["filepath"] and test["class_name"] == hit_case["class"] and test["method_name"] == hit_case["method"]:
                                    if k not in retrieved_tests:
                                        retrieved_tests[k] = {
                                            "repo": test["repo"],
                                            "revision": hit_case["revision"],
                                            "file_path": test["file_path"],
                                            "class_name": test["class_name"],
                                            "method_name": test["method_name"],
                                            "code": test["code"],
                                            "matched_vulns": []
                                        }
                                    if "vul" in hit_case:
                                        retrieved_tests[k]["matched_vulns"].append(hit_case["vul"])
    return retrieved_tests


if __name__ == "__main__":
    findings_df = pd.read_csv(os.path.join(INTHEWILD_FINDING_DIRPATH, INTHEWILD_FINDINGS_INSPECTED_FILENAME))
    confirmed_findings_df = findings_df[findings_df["final_decision"] == "Security Test"]
    findings_raw_results = os.path.join(INTHEWILD_FINDING_DIRPATH, "2", INTHEWILD_RESULTS_DIRNAME)
    findings_tests = extract_tests(findings_raw_results, confirmed_findings_df)
    print(f"Tests from Finding: {len(findings_tests)}")

    matchings_df = pd.read_csv(os.path.join(INTHEWILD_MATCHING_DIRPATH, INTHEWILD_MATCHINGS_INSPECTED_FILENAME))
    confirmed_matchings_df = matchings_df[matchings_df["final_decision"] == "Correct"]
    matchings_raw_results = os.path.join(INTHEWILD_MATCHING_DIRPATH, "e2e-lm", INTHEWILD_RESULTS_DIRNAME)
    all_tests = extract_tests(matchings_raw_results, confirmed_matchings_df, findings_tests)
    print(f"Tests from Matching: {len(all_tests) - len(findings_tests)}")

    with open(os.path.join(TEST4VUL_FILEPATH), 'w') as fout:
        json.dump(list(all_tests.values()), fout, indent=2)
