import json
import os
import subprocess
import tempfile

import git

from vuteco.core.common.constants import (CODESHOVEL_COMMAND,
                                          CODESHOVEL_JAR_FILEPATH)
from vuteco.core.common.utils_mining import (get_current_file_names_touched_in,
                                             retrieve_tests_from_file)


class FixCommitModel():
    history_cache: dict[str, dict[str, any]]

    def __init__(self) -> None:
        super().__init__()
        self.history_cache = {}

    def clear_cache(self) -> None:
        self.history_cache = {}

    def get_witnessing_score(self, repo_dir: str, file_path: str, method_name: str, fix_commits: list[git.Commit]) -> bool:
        return 1.0 if self.was_changed_in_fixes(repo_dir, file_path, method_name, fix_commits) else 0.0

    def was_changed_in_fixes(self, repo_dir: str, file_path: str, method_name: str, fix_commits: list[git.Commit]) -> bool:
        if repo_dir not in self.history_cache:
            self.history_cache[repo_dir] = {
                "fixes": [],
                "histories": {}
            }
        fix_to_process = [f for f in fix_commits if f.hexsha not in self.history_cache[repo_dir]["fixes"]]
        if len(fix_to_process) > 0:
            new_histories = self.get_modified_file_histories(repo_dir, fix_to_process)
            self.history_cache[repo_dir]["histories"].update(new_histories)
            self.history_cache[repo_dir]["fixes"].extend([f.hexsha for f in fix_to_process])
            #print("Update histories")
            #print(self.history_cache[repo_dir]["histories"])

        histories = self.history_cache[repo_dir]["histories"]
        if len(histories) == 0:
            return False
        fixes = [f.hexsha for f in fix_commits]
        for f_info in histories.values():
            for meth_hist in f_info:
                if "changeHistoryDetails" not in meth_hist:
                    continue
                for c_sha, c_info in meth_hist["changeHistoryDetails"].items():
                    if c_sha in fixes and method_name == c_info.get("functionName"):
                        #if c_info["type"] == "Yintroduced":
                        #print(f"Found! {method_name} in {test_case_filepath}")
                        #print(meth_hist["changeHistoryShort"])
                        #input()
                        return True
        return False

    def get_modified_file_histories(self, repo_dir: str, fix_commits: list[git.Commit]) -> dict[str, dict]:
        if fix_commits is None or len(fix_commits) == 0:
            return {}
        file_histories = {}
        for int_filepath in get_current_file_names_touched_in(fix_commits):
            #print(f"Looking for tests in {int_filepath}")
            for _, _, tm_node, _ in retrieve_tests_from_file(int_filepath):
                #print(f"Inspecting method {tm_node.name}")
                with tempfile.TemporaryDirectory() as tmp_dirname:
                    out_filepath = os.path.join(tmp_dirname, "results.json")
                    cs_command = CODESHOVEL_COMMAND.format(codeshovel_jar=CODESHOVEL_JAR_FILEPATH,
                                                           repo_dir=repo_dir,
                                                           file_path=os.path.relpath(int_filepath, repo_dir),
                                                           method_name=tm_node.name,
                                                           start_line=str(tm_node.position.line),
                                                           outfile_path=out_filepath).split(" ")
                    subprocess.call(cs_command, stdout=subprocess.DEVNULL)
                    if not os.path.exists(out_filepath):
                        continue
                    with open(out_filepath) as fin:
                        cs_output = json.load(fin)
                if int_filepath not in file_histories:
                    file_histories[int_filepath] = []
                file_histories[int_filepath].append(cs_output)
        return file_histories

    """
    def get_tests_changed(self, fix_commit: git.Commit) -> list[dict]:
        if fix_commit is None:
            return []
        tests = []
        changed_files = []
        for parent_commit in fix_commit.parents:
            for diff in fix_commit.diff(other=parent_commit, paths=[item for item in fix_commit.stats.files if ".java" in item], create_patch=True):
                changed_files.append({
                    "before_name": diff.b_path,
                    "after_name": diff.a_path,
                    "before_text": blob_to_text(diff.b_blob),
                    "after_text": blob_to_text(diff.a_blob),
                })
        for ch_file in changed_files:
            try:
                after_tests = retrieve_tests_from_class(ch_file["after_text"])
            except:
                print(f"Could not parse file {ch_file['after_name']}. Skipping it.")
                continue
            if ch_file['before_name'] is None:
                before_tests = []
            else:
                try:
                    before_tests = retrieve_tests_from_class(ch_file["before_text"])
                except:
                    print(f"Could not parse file {ch_file['before_name']}. Assuming no tests from it.")
                    before_tests = []
            for after_cu, after_class_node, after_method_node, after_test_code in after_tests:
                matched = False
                for _, _, _, before_test_code in before_tests:
                    if after_test_code == before_test_code:
                        matched = True
                if not matched:
                    class_fqn = get_class_fqn(after_cu, after_class_node)
                    tests.append({
                        "repo": fix_commit.repo.remotes.origin.url,
                        "commit": fix_commit.hexsha,
                        "file": ch_file["after_name"],
                        "class": class_fqn,
                        "method": after_method_node.name,
                        "code": after_test_code
                    })
        return tests
        """
