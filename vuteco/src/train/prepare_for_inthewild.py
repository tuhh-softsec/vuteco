import json
import os

import git
import pandas as pd
from core.common.constants import (INTHEWILD_FILEPATH,
                                   INTHEWILD_PREP_STATS_FILEPATH,
                                   REPOS_MERGE_FROM_TO, REPOS_TO_RENAME,
                                   TESTING_REPOS, VUL4J_FILEPATH,
                                   VUTECO_KB_FILEPATH)
from core.common.utils_mining import join_cwe_id_info, normalize_repo_url
from tqdm import tqdm

if __name__ == "__main__":
    with open(VUTECO_KB_FILEPATH) as fin:
        vuteco_kb: dict = json.load(fin)
    vul4j_df = pd.read_csv(VUL4J_FILEPATH)
    vul4j_cves = vul4j_df["cve_id"].tolist()
    stats = {
        "cvesInVul4J": 0,
        "cvesNoDescr": 0,
        "cvesNoCwe": 0,
        "cvesNoURLs": 0,
        "cvesAboutTesting": 0,
        "cvesNotClonable": 0,
        "cvesNoHead": 0,
        "projectAboutTesting": 0,
        "projectNotClonable": 0,
        "projectNoHead": 0
    }

    project_vulns: dict[str, dict] = {}
    for cve_id, info in vuteco_kb.items():
        if cve_id in vul4j_cves:
            stats["cvesInVul4J"] += 1
            continue
        if not info["description"]:
            stats["cvesNoDescr"] += 1
            continue
        if len(info["cwe_id"]) > 0:
            cwe_info = ",".join(join_cwe_id_info(c, n) for c, n in zip(info["cwe_id"], info["cwe_name"]))
        else:
            stats["cvesNoCwe"] += 1
            cwe_info = None
        fix_urls = [f["url"] for f in info["fixes"]]
        if len(fix_urls) == 0:
            stats["cvesNoURLs"] += 1
            continue
        url = normalize_repo_url(max(fix_urls, key=lambda x: fix_urls.count(x)))
        if url not in project_vulns:
            project_vulns[url] = {}
        project_vulns[url][cve_id] = {
            "description": info["description"],
            "cwe": cwe_info,
            "fixes": [f["hash"] for f in info["fixes"]]
        }

    inthewild_entries = []
    for url, cves in tqdm(project_vulns.items(), desc="Preparing input for VUTECO"):
        repo_name = os.path.join(*url.split("/")[-2:])
        if repo_name in TESTING_REPOS:
            stats["projectAboutTesting"] += 1
            stats["cvesAboutTesting"] += 1
            continue
        try:
            ls_remote_res: str = git.cmd.Git().ls_remote(url)
        except git.exc.GitCommandError:
            stats["projectNotClonable"] += 1
            stats["cvesNotClonable"] += len(cves.keys())
            continue
        #OLD_df = pd.read_csv("../data/inthewild/OLD_inthewild.csv")
        #head_rev = OLD_df.loc[OLD_df['url'] == url, 'revision'].iloc[0]
        # Everytime we run this, we get the most recent HEAD
        head_rev = ls_remote_res.splitlines()[0].split('\t')[0].strip()
        if not head_rev:
            stats["projectNoHead"] += 1
            stats["cvesNoHead"] += len(cves.keys())
            continue
        inthewild_entries.append({
            "url": url,
            "revision": head_rev,
            "vuln_ids": list(cves.keys()),
            # "fixes": ",".join(list(dict.fromkeys([f for c in cves.values() for f in c["fixes"]])))
            # "vuln_descriptions": ",".join(list(cves.values())),
        })

    for entry in inthewild_entries:
        repo_name = os.path.join(*entry["url"].split("/")[-2:])
        for k in REPOS_MERGE_FROM_TO:
            if k == repo_name:
                entry["url"] = f'https://github.com/{REPOS_MERGE_FROM_TO[k]}'
                break
        for k in REPOS_TO_RENAME:
            if k == repo_name:
                entry["url"] = f'https://github.com/{REPOS_TO_RENAME[k]}'
                break
    inthewild_df = pd.DataFrame(inthewild_entries)
    inthewild_df = inthewild_df.groupby(['url', "revision"], sort=False)["vuln_ids"].sum().reset_index()
    inthewild_df["vuln_ids"] = inthewild_df["vuln_ids"].apply(lambda x: ','.join(x))
    inthewild_df.to_csv(INTHEWILD_FILEPATH, index=False)
    stats["nrProjects"] = inthewild_df["url"].nunique()
    stats["nrVulns"] = len(set(','.join(inthewild_df["vuln_ids"]).split(",")))
    with open(INTHEWILD_PREP_STATS_FILEPATH, "w") as fout:
        json.dump(stats, fout, indent=2)
