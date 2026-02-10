import csv
import json
import os
import tarfile
import tempfile
import traceback

import docker
from halo import Halo
from tqdm import tqdm

from vuteco.core.common.constants import (DATA_DIRPATH, IS_SECURITY_COL,
                                          VUL4J_DIRPATH, VUL4J_FILEPATH,
                                          VUL4J_TEST_FILEPATH)
from vuteco.core.common.utils_mining import (get_class_fqn, get_cve_descr_date,
                                             get_java_files_in_dir,
                                             retrieve_tests_from_file)

if __name__ == "__main__":
    try:
        DOCKER_CLIENT = docker.from_env(timeout=300)
    except Exception:
        print(f"Error while instantiating the Docker SDK client. Likely, Docker daemon is not running, so please turn it on. Here the detail:")
        traceback.print_exc()
        exit(1)

    if not os.path.exists(VUL4J_FILEPATH):
        print("Vul4j dataset not found. Exiting.")
        exit(1)
    with open(VUL4J_FILEPATH) as fin:
        vul4j_entries = [dict(row) for row in csv.DictReader(fin)]
    # Parse the "failing_tests" column as list
    for v4je in vul4j_entries:
        v4je["failing_tests"] = v4je["failing_tests"].split(",") if "," in v4je["failing_tests"] else [v4je["failing_tests"]]

    if os.path.exists(VUL4J_TEST_FILEPATH):
        with open(VUL4J_TEST_FILEPATH) as fin:
            vul4j_tests: dict = json.load(fin)
    else:
        vul4j_tests = {}

    base_repopath = "/tmp/vul4j/{}"
    base_checkout_cmd = "vul4j checkout --id {} -d {}"
    base_move_cmd = "cat {}"
    base_all_commands = f'/bin/sh -c "{"{} ; {}"}"'

    for idx, v4je in enumerate(vul4j_entries):
        vul_id = v4je["vul_id"]
        # if vul_id in vul4j_tests:
        #    continue
        try:
            cve_desc = get_cve_descr_date(v4je["cve_id"])
        except:
            print(f"Error retrieving data for {v4je['cve_id']} from the API")
            cve_desc = None
        vul4j_tests[vul_id] = {
            "repo": v4je["repo_slug"],
            "revision": os.path.basename(os.path.normpath(v4je["human_patch"])),
            "cve": v4je["cve_id"],
            "cve_desc": cve_desc,
            "cwe": v4je["cwe_id"],
            "cwe_name": v4je["cwe_name"],
            "tests": []
        }
        repopath = base_repopath.format(vul_id)
        checkout_cmd = base_checkout_cmd.format(vul_id, repopath)
        print(f"({idx+1}/{len(vul4j_entries)}) Collecting tests from '{v4je['repo_slug']}'")
        module = v4je['failing_module'] if v4je['failing_module'] != "root" else os.path.curdir
        # if "groovy" in test_dir:
        #    print("Skipping .groovy tests")
        #    continue

        vul4j_failing_tests = []
        for sectest_case in v4je['failing_tests']:
            sectest_class_name, sectest_method_name = sectest_case.split("#")
            if "[" in sectest_method_name:
                sectest_method_name = sectest_method_name.split("[")[0]
            vul4j_failing_tests.append({
                "file": os.path.relpath(os.path.join(module, v4je["test"], *sectest_class_name.split(".")) + f".java"),
                "class": sectest_class_name,
                "method": sectest_method_name
            })
        # print(vul4j_failing_tests)
        # continue

        # Copy the repo from the container into a temporary local directory, to make things easier
        temp_dir = tempfile.TemporaryDirectory()
        with Halo(text="Booting Vul4J container") as spinner:
            try:
                container = DOCKER_CLIENT.containers.run(image="bqcuongas/vul4j", command=checkout_cmd, detach=True)
                container.wait()
                bits, _ = container.get_archive(repopath)
                with open(os.path.join(temp_dir.name, "export.tar"), 'wb') as outfile:
                    for d in bits:
                        outfile.write(d)
                tar = tarfile.open(os.path.join(temp_dir.name, "export.tar"))
                tar.extractall(path=temp_dir.name, numeric_owner=True)
                tar.close()
            except Exception as e:
                traceback.print_exc()
                exit(1)
            finally:
                container.stop()
                container.remove(force=True)
            spinner.succeed("Vul4J container booted!")
        local_repo_dir = os.path.join(temp_dir.name, vul_id)

        # Loop over all Java files in the project, but only those in the same module of the security test
        module_test_dir = os.path.abspath(os.path.join(local_repo_dir, module, v4je["test"]))
        java_files = get_java_files_in_dir(module_test_dir)
        for jf_path in tqdm(java_files, desc=f"Inspecting {len(java_files)} Java files"):
            jf_relpath = os.path.relpath(jf_path, local_repo_dir)
            try:
                tests = retrieve_tests_from_file(jf_path)
            except Exception as e:
                print(f"Could not read file {jf_path} correctly. Skipping it.")
                continue
            for jf_cu, class_node, tm_node, test_code in tests:
                class_fqn = get_class_fqn(jf_cu, class_node)
                dest_file = os.path.join(VUL4J_DIRPATH, vul_id, class_fqn, f"{tm_node.name}.txt")
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                with open(dest_file, 'w') as fout:
                    fout.write(test_code)
                test_dict = {
                    "originatingFile": jf_relpath,
                    "class": class_fqn,
                    "method": tm_node.name,
                    "codeFile": os.path.relpath(dest_file, DATA_DIRPATH),
                    IS_SECURITY_COL: next((ft for ft in vul4j_failing_tests if ft["file"] == jf_relpath and ft["class"]
                                          == class_fqn and ft["method"] == tm_node.name), None) is not None
                }
                vul4j_tests[vul_id]["tests"].append(test_dict)
        temp_dir.cleanup()
        with open(VUL4J_TEST_FILEPATH, "w") as fout:
            json.dump(vul4j_tests, fout, indent=2)
