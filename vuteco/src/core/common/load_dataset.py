import itertools
import json
import os

import pandas as pd
from core.common.constants import (CODE_COL, CVE_DESCR_COL, CVE_ID_COL,
                                   CWE_INFO_COL, IS_SECURITY_COL, LABEL_COL,
                                   TEXT_1_COL, TEXT_2_1_COL, TEXT_2_COL,
                                   TEXT_COL)
from core.common.utils_mining import join_cwe_id_info


def find_duplicates(vul4j_df: pd.DataFrame, dup_col):
    all_to_drop = []
    for _, group_df in vul4j_df.groupby(dup_col):
        if len(group_df) > 1:
            #print(group_df)
            #input()
            to_drop = group_df.index.tolist()
            if (group_df[LABEL_COL] == 1).any():
                to_drop.remove(group_df.index[group_df[LABEL_COL] == 1][0])
                all_to_drop.extend(to_drop)
            else:
                to_drop.remove(to_drop[0])
                all_to_drop.extend(to_drop)
    return all_to_drop


def make_pairs(df: pd.DataFrame, valid_pair_df: pd.DataFrame):
    pairs = []
    set_1 = df[TEXT_1_COL].unique()
    set_2 = df[TEXT_2_COL].unique()
    for pair in itertools.product(set_1, set_2):
        query = valid_pair_df[(valid_pair_df[TEXT_1_COL] == pair[0]) & (valid_pair_df[TEXT_2_COL] == pair[1])].to_dict(orient="records")
        cwes = valid_pair_df[valid_pair_df[TEXT_2_COL] == pair[1]][TEXT_2_1_COL].unique().tolist()
        if len(query) == 0:
            pairs.append({
                TEXT_1_COL: pair[0],
                TEXT_2_COL: pair[1],
                TEXT_2_1_COL: ",".join(c for c in cwes if c is not None) if cwes else None,
                LABEL_COL: 0,
            })
        else:
            for q in query:
                pairs.append({
                    TEXT_1_COL: q[TEXT_1_COL],
                    TEXT_2_COL: q[TEXT_2_COL],
                    TEXT_2_1_COL: ",".join(c for c in cwes if c is not None) if cwes else None,
                    LABEL_COL: 1,
                })
    return pd.DataFrame(pairs)


def load_vul4j(add_test_code: bool, add_cve_info: bool, add_metadata: bool, dataset_filepath: str, dataset_basepath: str, stop_after_loading: int = None):
    if not os.path.exists(dataset_filepath) or not os.path.exists(dataset_basepath):
        print(f"Dataset file or directory containing Vul4J dataset not found.")
        return None
    with open(dataset_filepath) as fin:
        vul4j_tests: dict = json.load(fin)
    vul4j_tests_list = []
    for vul4j_id, v in vul4j_tests.items():
        for test in v["tests"]:
            with open(os.path.join(dataset_basepath, test["codeFile"])) as fin:
                test_code = fin.read()
            entry = {}
            if add_test_code:
                entry[CODE_COL] = test_code
            if add_cve_info:
                if v["cwe"] is None or v["cwe"] == "" or v["cwe"] == "Not Mapping":
                    cwe_info = None
                else:
                    cwe = v["cwe"]
                    cwe_name = v["cwe_name"]
                    if type(cwe) is list and type(cwe_name) is list:
                        cwe_info = ",".join(join_cwe_id_info(c, n) for c, n in zip(cwe, cwe_name))
                    else:
                        cwe_info = join_cwe_id_info(cwe, cwe_name)
                entry[CWE_INFO_COL] = cwe_info
                entry[CVE_ID_COL] = v["cve"]
                entry[CVE_DESCR_COL] = v["cve_desc"]
            if add_metadata:
                entry["file"] = test["originatingFile"]
                entry["class"] = test["class"]
                entry["method"] = test["method"]
                # All vul4j projects are from github, so we can safely use github.com
                entry["url"] = f"https://github.com/{v['repo']}"
                entry["fix"] = v["revision"] if ".." not in v["revision"] else v["revision"].split("..")[1]
            entry[IS_SECURITY_COL] = test[IS_SECURITY_COL]
            vul4j_tests_list.append(entry)
        if stop_after_loading is not None and len(vul4j_tests_list) >= stop_after_loading:
            break
        print(f"- {vul4j_id}: Loaded {len(v['tests'])} tests", flush=True)
    vul4j_df = pd.DataFrame(vul4j_tests_list)
    vul4j_df[LABEL_COL] = vul4j_df[IS_SECURITY_COL].astype(int)
    vul4j_df = vul4j_df.drop(columns=[IS_SECURITY_COL])
    return vul4j_df


def load_vul4j_for_fnd(dataset_filepath: str, dataset_basepath: str, stop_after_loading: int = None) -> pd.DataFrame:
    vul4j_df = load_vul4j(True, False, True, dataset_filepath, dataset_basepath, stop_after_loading)
    vul4j_df[TEXT_COL] = vul4j_df[CODE_COL]
    vul4j_df = vul4j_df.drop(columns=[CODE_COL])
    vul4j_df = vul4j_df.drop(find_duplicates(vul4j_df, TEXT_COL), axis="index").reset_index(drop=True)
    return vul4j_df


def load_vul4j_for_lnk(dataset_filepath: str, dataset_basepath: str, stop_after_loading: int = None) -> pd.DataFrame:
    vul4j_df = load_vul4j(True, True, False, dataset_filepath, dataset_basepath, stop_after_loading)
    vul4j_df[TEXT_1_COL] = vul4j_df[CODE_COL]
    vul4j_df[TEXT_2_COL] = vul4j_df[CVE_DESCR_COL]
    vul4j_df[TEXT_2_1_COL] = vul4j_df[CWE_INFO_COL]
    vul4j_df = vul4j_df.drop(columns=[CODE_COL, CVE_DESCR_COL, CWE_INFO_COL])
    # Not necessary to drop duplicates for the Linker, but better safe than sorry
    vul4j_df = vul4j_df.drop(find_duplicates(vul4j_df, TEXT_1_COL), axis="index").reset_index(drop=True)
    valid_df = vul4j_df[vul4j_df[LABEL_COL] == 1].dropna(subset=[TEXT_1_COL, TEXT_2_COL])
    pairs_df = make_pairs(valid_df, valid_df)
    cve_df = valid_df.drop([LABEL_COL, TEXT_1_COL, TEXT_2_1_COL], axis=1).drop_duplicates()
    return pd.merge(pairs_df, cve_df, on=[TEXT_2_COL], how="left")


def load_vul4j_for_e2e(dataset_filepath: str, dataset_basepath: str, stop_after_loading: int = None) -> pd.DataFrame:
    vul4j_df = load_vul4j(True, True, True, dataset_filepath, dataset_basepath, stop_after_loading)
    vul4j_df[TEXT_1_COL] = vul4j_df[CODE_COL]
    vul4j_df[TEXT_2_COL] = vul4j_df[CVE_DESCR_COL]
    vul4j_df[TEXT_2_1_COL] = vul4j_df[CWE_INFO_COL]
    vul4j_df = vul4j_df.drop(columns=[CODE_COL, CVE_DESCR_COL, CWE_INFO_COL])  # , "class", "method", "fix"])
    vul4j_df = vul4j_df.drop(find_duplicates(vul4j_df, TEXT_1_COL), axis="index").reset_index(drop=True)
    # Make pair only if originate from the same project
    all_pairs = []
    for g_name, g_df in vul4j_df.groupby("url", sort=False):
        g_valid_df = g_df[g_df[LABEL_COL] == 1].dropna(subset=[TEXT_1_COL, TEXT_2_COL])
        group_pairs_df = make_pairs(g_df, g_valid_df).drop_duplicates().dropna(subset=[TEXT_1_COL, TEXT_2_COL])
        code_df = g_df[[TEXT_1_COL, "file", "class", "method"]].drop_duplicates().dropna()
        cve_df = g_df[[TEXT_2_COL, CVE_ID_COL]].drop_duplicates().dropna()
        group_pairs_df = pd.merge(group_pairs_df, code_df, on=[TEXT_1_COL], how="right").dropna(subset=[TEXT_1_COL, TEXT_2_COL, LABEL_COL])
        group_pairs_df = pd.merge(group_pairs_df, cve_df, on=[TEXT_2_COL], how="left")
        group_pairs_df["url"] = g_name
        fixes = list(dict.fromkeys(vul4j_df[vul4j_df["url"] == g_name]["fix"].tolist()))
        group_pairs_df["fixes"] = [fixes] * len(group_pairs_df)
        all_pairs.append(group_pairs_df)
    total_df = pd.concat(all_pairs)
    total_df[LABEL_COL] = total_df[LABEL_COL].astype(int)
    return total_df


def load_vul4j_for_heu(dataset_filepath: str, dataset_basepath: str, stop_after_loading: int = None) -> pd.DataFrame:
    return load_vul4j(False, False, True, dataset_filepath, dataset_basepath, stop_after_loading)
