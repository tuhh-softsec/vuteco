import datetime as dt
import tempfile

import datasets
import numpy as np
from git import Repo

from vuteco.core.common.constants import LABEL_COL
from vuteco.core.common.utils_training import (compute_performance_clf,
                                               print_stdout_file)
from vuteco.core.modeling.modeling_fix import FixCommitModel


def row_to_tuple(row):
    return (row["url"], row["file"], row["class"], row["method"])


def eval_fix(test_ds: datasets.Dataset, log_outfile: str) -> dict:
    fix_commit_model = FixCommitModel()
    url_and_fixes: dict[str, list] = {}
    for row in test_ds:
        if row["url"] not in url_and_fixes:
            url_and_fixes[row["url"]] = []
        curr_fixes = row["fixes"] if "fixes" in row else [row["fix"]] if "fix" in row else []
        url_and_fixes[row["url"]] = list(dict.fromkeys(url_and_fixes[row["url"]] + curr_fixes))
    flagged_tests = []
    for idx, (url, fixes) in enumerate(url_and_fixes.items()):
        if url is None or fixes is None or len(fixes) == 0:
            print_stdout_file(f"The clone URL or the fix commit is missing for entry nr. {idx} in the test dataset. Skipping.", log_outfile)
            continue
        print_stdout_file(f"[{dt.datetime.now()}] ({idx+1}/{len(url_and_fixes)}) Cloning from {url}", log_outfile)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            try:
                repo = Repo.clone_from(url, tmp_dirname)
            except:
                print_stdout_file(f"Failed to clone from {url}", log_outfile)
                continue
            tests = []
            fix_commits = []
            for f in fixes:
                try:
                    fix_commits.append(repo.commit(f))
                except:
                    pass
            print_stdout_file(f"[{dt.datetime.now()}] ({idx+1}/{len(url_and_fixes)}) Checking among the modified tests", log_outfile)
            for f_row in test_ds.filter(lambda x: x["url"] == url):
                if fix_commit_model.was_changed_in_fixes(repo.working_dir, f_row["file"], f_row["method"], fix_commits):
                    tests.append((url, f_row["file"], f_row["class"], f_row["method"]))
            fix_commit_model.clear_cache()
            print_stdout_file(f"[{dt.datetime.now()}] Retrieved {len(tests)} tests from fix commits {fixes}", log_outfile)
            flagged_tests.extend(tests)
    retrieved_tests = set(flagged_tests)
    print_stdout_file(f"[{dt.datetime.now()}] Flagged a total of {len(retrieved_tests)} tests", log_outfile)
    # relevant_tests = {row_to_tuple(row) for row in test_ds.filter(lambda x: x[LABEL_COL] == 1)}
    # non_relevant_tests = {row_to_tuple(row) for row in test_ds.filter(lambda x: x[LABEL_COL] == 0)}
    # test_results = compute_performance_for_cb(relevant_tests, non_relevant_tests, retrieved_tests)
    predictions = [1 if row_to_tuple(row) in retrieved_tests else 0 for row in test_ds]
    test_results = compute_performance_clf(test_ds[LABEL_COL], predictions)
    return {
        "model_config": {},
        "test_results": {
            "test_performance": {k: None if np.isnan(v) else v for k, v in test_results.items()},
            "test_predictions": predictions
        }
    }
