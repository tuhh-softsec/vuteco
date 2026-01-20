import json
import os
from collections import Counter
from statistics import mean, median

import pandas as pd
from common.constants import (DATA_DIRPATH, INTHEWILD_FILEPATH,
                              INTHEWILD_FINDING_DIRPATH, INTHEWILD_FINDINGS_DIRNAME,
                              INTHEWILD_FINDINGS_TO_INSPECT_FILENAME,
                              INTHEWILD_MATCHES_DIRNAME, INTHEWILD_MATCHING_DIRPATH,
                              INTHEWILD_MATCHINGS_TO_INSPECT_FILENAME,
                              INTHEWILD_RESULTS_DIRNAME, INTHEWILD_STATS_FILENAME,
                              INTHEWILD_TESTS_DIRNAME,
                              INTHEWILD_TOTAL_STATS_FILENAME, LABEL_COL,
                              REPOS_MERGE_FROM_TO, REPOS_TO_RENAME,
                              TESTING_REPOS, VUL4J_TEST_FILEPATH,
                              VUTECO_KB_FILEPATH, TechniqueName)
from common.load_dataset import load_vul4j_for_fnd


def process_revision_project(rev_type: str, repo: str, vulns: list, rev_id: str, rev_info: dict, techniques: list, cutoffs: dict):
    vulns_info = {
        "vulns": ",".join(vulns),
        "nrVulns": len(vulns),
        "nrMatches": len(vulns) * rev_info["stats"]["nrTests"]
    } if vulns is not None else {}
    revision_stats = {
        "repo": repo,
        "revision": rev_id,
        "type": rev_type,
        "nrTestsTotal": rev_info["stats"]["nrTests"],
        **vulns_info,
        **{f"nrTestsFlagged@{k}": 0 for k in cutoffs.keys()},
        **{f"nrTestsFlagged@{k}_{t}": 0 for k in cutoffs.keys() for t in techniques},
    }
    revision_tests = {}
    for test_id, test_info in rev_info["judgments"].items():
        test_key = f'{repo}_{rev_id}_{test_id}'
        if test_key not in revision_tests:
            revision_tests[test_key] = {
                "repo": repo,
                "revision": rev_id,
                "type": rev_type,
                "filepath": test_info["file_path"],
                "class": test_info["class_name"],
                "method": test_info["method_name"],
                "startline": test_info["startline"],
                **{t: [] for t in techniques}
            }
        revision_flagged_tests_at_k = {k: set() for k in cutoffs.keys()}
        for tech in techniques:
            # "matched_vulns" has the precedence
            if "matched_vulns" in test_info:
                if tech in test_info["matched_vulns"]:
                    for vul, score in test_info["matched_vulns"][tech].items():
                        revision_tests[test_key][tech].append({
                            "vul": vul,
                            "score": score
                        })
                        for k, v in cutoffs.items():
                            if score > v:
                                revision_flagged_tests_at_k[k].add(test_key)
            elif "witnessing_scores" in test_info:
                if tech in test_info["witnessing_scores"]:
                    score = test_info["witnessing_scores"][tech]
                    revision_tests[test_key][tech].append({
                        "score": score
                    })
                    for k, v in cutoffs.items():
                        if score > v:
                            revision_flagged_tests_at_k[k].add(test_key)
            if len(revision_tests[test_key][tech]) > 0:
                for k, v in cutoffs.items():
                    if score > v:
                        revision_stats[f"nrTestsFlagged@{k}_{tech}"] += 1
        for k, the_set in revision_flagged_tests_at_k.items():
            revision_stats[f"nrTestsFlagged@{k}"] = len(the_set)
    return revision_tests, revision_stats


def extract_results(main_dir: str, second_dir: str, techniques: list, drop_dup_repos: bool=False):
    raw_results_dirpath = os.path.join(main_dir, INTHEWILD_RESULTS_DIRNAME)
    stats_dirpath = os.path.join(main_dir, INTHEWILD_STATS_FILENAME)
    second_dirpath = os.path.join(main_dir, second_dir)
    tests_dirpath = os.path.join(main_dir, INTHEWILD_TESTS_DIRNAME)
    all_revision_stats = []
    all_tests = {}
    if not os.path.exists(raw_results_dirpath):
        print(f"Main directory {raw_results_dirpath} not found")
        return
    all_cutoffs = {
        "Any": 0.0,
        "10": 0.10,
        "25": 0.25,
        "50": 0.50,
        "75": 0.75,
        "80": 0.80,
        "90": 0.90,
    }
    cutoffs = {
        "50": 0.50,
        "80": 0.80,
        "90": 0.90,
    }
    inthewild_df = pd.read_csv(INTHEWILD_FILEPATH)
    inthewild_df["repo"] = inthewild_df["url"].apply(lambda x: os.path.join(*x.split("/")[-2:]))
    inthewild_df = inthewild_df.drop(["url"], axis=1)
    for root, _, files in os.walk(raw_results_dirpath):
        for f in files:
            with open(os.path.join(root, f)) as fin:
                project_results = json.load(fin)
            vuln_ids = list(project_results["candidate_vulns"].keys()) if "candidate_vulns" in project_results else None
            for rev_id, rev_info in project_results["revisions"].items():
                if project_results["repo"] in TESTING_REPOS:
                    continue
                if project_results["repo"] in REPOS_MERGE_FROM_TO.keys():
                    if drop_dup_repos:
                        continue
                    else:
                        project_results["repo"] = REPOS_MERGE_FROM_TO[project_results["repo"]]
                if project_results["repo"] in REPOS_TO_RENAME:
                    project_results["repo"] = REPOS_TO_RENAME[project_results["repo"]]
                revision_tests, revision_stats = process_revision_project("HEAD", project_results["repo"], vuln_ids, rev_id, rev_info, techniques, all_cutoffs)
                all_tests.update(revision_tests)
                all_revision_stats.append(revision_stats)
            # if "revisions_after_patch" in project_results:
            #     for vul_id, vul_info in project_results["revisions_after_patch"].items():
            #         revision_tests, revision_stats = process(
            #             "After-Patch", project_results["repo"], [vul_id], vul_info["last_patch"], {"stats": vul_info["stats"], "judgments": vul_info["judgments"]}, techniques)
            #         all_tests.update(revision_tests)
            #         all_revision_stats.append(revision_stats)
        revision_stats_df = pd.DataFrame(all_revision_stats)
        revision_stats_df.to_csv(stats_dirpath, index=False)
    merged_df = revision_stats_df.merge(inthewild_df)
    merged_df["nrVulns"] = merged_df["vuln_ids"].apply(lambda x: len(x.split(",")))
    merged_df["nrPairsTestsVulns"] = merged_df["nrTestsTotal"] * merged_df["nrVulns"]
    total_stats = {
        "nrProjectsWithTests": int(len(merged_df[merged_df["nrTestsTotal"] > 0])),
        "nrTestsTotal": int(merged_df["nrTestsTotal"].sum()),
        "nrVulnsTotal": int(merged_df["nrVulns"].sum()),
        "nrVulnsInProjectWithTests": int(merged_df[merged_df["nrTestsTotal"] > 0]["nrVulns"].sum()),
        "nrPairsTestsVulnsTotal": int(merged_df["nrPairsTestsVulns"].sum())
    }
    with open(os.path.join(main_dir, INTHEWILD_TOTAL_STATS_FILENAME), "w") as fout:
        json.dump(total_stats, fout, indent=2)

    with open(VUTECO_KB_FILEPATH) as fin:
       knowledge_base: dict = json.load(fin)

    fnd_df = load_vul4j_for_fnd(VUL4J_TEST_FILEPATH, os.path.abspath(DATA_DIRPATH))
    vul4j_sec_tests_df = fnd_df[fnd_df[LABEL_COL] == 1].copy()
    del fnd_df
    vul4j_sec_tests_df["repo"] = vul4j_sec_tests_df["url"].apply(lambda x: os.path.join(*x.split("/")[-2:]))
    vul4j_sec_tests_df["filepath"] = vul4j_sec_tests_df["file"]
    vul4j_sec_tests_df = vul4j_sec_tests_df.drop(["url", "file"], axis=1)
    all_tests_df = pd.DataFrame(all_tests.values())
    for t in techniques:
        tech_tests_df: pd.DataFrame = all_tests_df[all_tests_df[t].apply(len) > 0].drop(columns=[c for c in techniques if c != t]).reset_index(drop=True)
        for k, thres in cutoffs.items():
            k_tech_tests_df = tech_tests_df.copy()
            print(f"Thresholds {k} ({thres}):")
            k_tech_tests_df[t] = k_tech_tests_df[t].map(
                lambda row: [dct for dct in row if dct["score"] > thres]
            )
            k_tech_tests_df = k_tech_tests_df[k_tech_tests_df[t].apply(len) > 0].reset_index(drop=True)
            
            # # Adding CWE information
            # k_tech_tests_df[t] = k_tech_tests_df[t].map(lambda row: [
            #     {**dct, "cwe": knowledge_base[dct["vul"]]["cwe_id"] if dct["vul"] in knowledge_base else None} if "vul" in dct else dct
            #     for dct in row
            # ])
            if len(k_tech_tests_df[t].tolist()) > 0 and "vul" in k_tech_tests_df[t].tolist()[0][0]:
                k_tech_tests_df["nrMatches"] = k_tech_tests_df[t].apply(len)
                matched_vulns = [x["vul"] for lst in k_tech_tests_df[t] for x in lst if "vul" in x]
                vulns_nr_tests = Counter(matched_vulns)
                print(f"- {t} matched with {len(vulns_nr_tests.keys())} distinct vulnerabilities")
                print(f"- {t} matched an average of {mean(vulns_nr_tests.values())} tests to each vulnerability")
                print(f"- {t} matched a median of {median(vulns_nr_tests.values())} tests to each vulnerability")
            
            k_tech_tests_df = k_tech_tests_df.dropna(axis=1, how="all")
            if t in k_tech_tests_df.columns:
                # We ignore the sec. test cases appearing in Vul4J
                tests_to_drop_df = k_tech_tests_df.merge(vul4j_sec_tests_df[["repo", "filepath", "method"]], on=["repo", "filepath", "method"], how='left', indicator=True)
                k_tech_tests_df = tests_to_drop_df[tests_to_drop_df['_merge'] == 'left_only'].drop(columns=['_merge'])
                # TODO Call compute_grouped_estimations() for the new metrics
                # Export phase
                k_tech_tests_df = k_tech_tests_df.rename(columns={t: "scores"})
                os.makedirs(tests_dirpath, exist_ok=True)
                k_tech_tests_df.to_csv(os.path.join(tests_dirpath, f"{t}_{k}.csv"), index=False)
                tech_tests_exploded = k_tech_tests_df.explode("scores", ignore_index=True)
                matches_df = pd.concat([
                    tech_tests_exploded.drop(columns=["nrMatches", "scores"], errors='ignore'),
                    pd.json_normalize(tech_tests_exploded["scores"])
                ], axis=1)
                os.makedirs(second_dirpath, exist_ok=True)
                matches_df.to_csv(os.path.join(second_dirpath, f"{t}_{k}.csv"), index=False)
                if "cwe" in matches_df:
                    print(f"- Nr. CWEs matched: {matches_df[matches_df['score'] >= thres]['cwe'].value_counts()}")


if __name__ == "__main__":
    #extract_results(os.path.join(INTHEWILD_FINDING_DIRPATH, "1"), INTHEWILD_FINDINGS_DIRNAME, [TechniqueName.CODELLAMA, TechniqueName.DEEPSEEKCODER, TechniqueName.CODET5PLUS, TechniqueName.FIX], drop_dup_repos=True)
    extract_results(os.path.join(INTHEWILD_FINDING_DIRPATH), INTHEWILD_FINDINGS_DIRNAME, [TechniqueName.UNIXCODER], drop_dup_repos=True)
    selected_finding_file = os.path.join(INTHEWILD_FINDING_DIRPATH, INTHEWILD_FINDINGS_DIRNAME, f"{TechniqueName.UNIXCODER}_50.csv")
    findings_df = pd.read_csv(selected_finding_file).drop("type", axis=1)
    findings_df = findings_df[~findings_df["repo"].isin(TESTING_REPOS)]
    findings_df = findings_df[~findings_df["repo"].isin(REPOS_MERGE_FROM_TO.keys())]
    findings_df.insert(0, "final_decision", "")
    findings_df.insert(0, "sec_sec", "")
    findings_df.insert(0, "sec_notsec", "")
    findings_df.insert(0, "notsec_sec", "")
    findings_df.insert(0, "notsec_notsec", "")
    findings_df.insert(0, "agree", "")
    findings_df.insert(0, "inspector_2", "")
    findings_df.insert(0, "inspector_1", "")
    findings_df.insert(0, "blame_url", "https://github.com/" + findings_df["repo"] + "/blame/" + findings_df["revision"] + "/" + findings_df["filepath"] + "#L" + findings_df["startline"].astype(str))
    findings_df.to_csv(os.path.join(INTHEWILD_FINDING_DIRPATH, INTHEWILD_FINDINGS_TO_INSPECT_FILENAME), index=False)

    #extract_results(os.path.join(INTHEWILD_MATCHING_DIRPATH, "e2e-nn"), INTHEWILD_MATCHES_DIRNAME, [TechniqueName.CODEBERT, TechniqueName.CODET5PLUS, TechniqueName.UNIXCODER, TechniqueName.FIX])
    extract_results(os.path.join(INTHEWILD_MATCHING_DIRPATH), INTHEWILD_MATCHES_DIRNAME, [TechniqueName.DEEPSEEKCODER])
    selected_matching_file = os.path.join(INTHEWILD_MATCHING_DIRPATH, INTHEWILD_MATCHES_DIRNAME, f"{TechniqueName.DEEPSEEKCODER}_50.csv")
    matchings_df = pd.read_csv(selected_matching_file).drop("type", axis=1)
    matchings_df['repo'] = matchings_df['repo'].replace(REPOS_MERGE_FROM_TO)
    matchings_df = matchings_df.drop_duplicates(subset=['repo', "filepath", "class", "method", 'vul'], keep='first')
    matchings_df = matchings_df[~matchings_df["repo"].isin(TESTING_REPOS)]
    matchings_df.insert(0, "final_decision", "")
    matchings_df.insert(0, "correct_correct", "")
    matchings_df.insert(0, "correct_incorrect", "")
    matchings_df.insert(0, "incorrect_correct", "")
    matchings_df.insert(0, "incorrect_incorrect", "")
    matchings_df.insert(0, "agree", "")
    matchings_df.insert(0, "inspector_2", "")
    matchings_df.insert(0, "inspector_1", "")
    matchings_df.insert(0, "blame_url", "https://github.com/" + matchings_df["repo"] + "/blame/" + matchings_df["revision"] + "/" + matchings_df["filepath"] + "#L" + matchings_df["startline"].astype(str))
    matchings_df.to_csv(os.path.join(INTHEWILD_MATCHING_DIRPATH, INTHEWILD_MATCHINGS_TO_INSPECT_FILENAME), index=False)
