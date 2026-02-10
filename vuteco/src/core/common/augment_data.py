import os
import re
import subprocess
import tempfile
from enum import Enum
from random import randint, seed
from typing import Union

import datasets
import pandas as pd
from core.common.constants import (DETERMINISM, JT_COMMAND, JT_JAR_FILEPATH,
                                   JT_RULES, LABEL_COL, RANDOM_SEED,
                                   SPAT_COMMAND, SPAT_JAR_FILEPATH,
                                   SPAT_JVM_PATHS, SPAT_PREFIX, SPAT_SUFFIX,
                                   TEXT_1_COL, TEXT_2_1_COL, TEXT_2_COL,
                                   TEXT_COL)
from core.common.utils_training import print_stdout_file
from imblearn.over_sampling import RandomOverSampler


class AugmentationApproach(str, Enum):
    NONE = "none"
    JT = "jt"
    SPAT = "spat"
    BOOTSTRAP = "bs"


def get_random_long_seed():
    return randint(-(2**63), (2**63) - 1)


def copy_dataset(input_ds: datasets.Dataset):
    return datasets.Dataset.from_pandas(pd.DataFrame(input_ds))


def find_java_runtime_jar():
    for jvm_path in SPAT_JVM_PATHS:
        rt_jars = [os.path.join(root, file) for root, _, files in os.walk(jvm_path) for file in files if re.compile(r"((rt)|(jrt-fs))\.jar").match(file)]
        if len(rt_jars) > 0:
            return rt_jars[0]
    return None


def call_spat(rule_id: str, in_temp_dir: str, rt_jar_filepath: str, seed: int, log_outfile: str = None):
    new_instances = []
    out_temp_dir = tempfile.TemporaryDirectory()
    spat_rule_command = SPAT_COMMAND.format(spat_jar=SPAT_JAR_FILEPATH, seed=seed, rule_id=rule_id, in_dir=in_temp_dir, out_dir=out_temp_dir.name, rt_jar=rt_jar_filepath).split(" ")
    if log_outfile and os.path.exists(log_outfile):
        with open(log_outfile, "a") as fout:
            subprocess.call(spat_rule_command, stdout=subprocess.DEVNULL, stderr=fout)
    else:
        subprocess.call(spat_rule_command, stdout=subprocess.DEVNULL)

    out_java_filepaths = [os.path.join(root, file) for root, _, files in os.walk(out_temp_dir.name) for file in files if re.compile(r".*\.java").match(file)]
    print_stdout_file(f"- SPAT's rule #{rule_id} generated {len(out_java_filepaths)} new files", log_outfile)
    for out_java_filepath in out_java_filepaths:
        with open(out_java_filepath) as fin:
            new_text = fin.read().removeprefix(SPAT_PREFIX).removesuffix(SPAT_SUFFIX).strip()
        file_id = os.path.basename(out_java_filepath).split(".")[0]
        text_2_filepath = os.path.join(in_temp_dir, f"{file_id}.cve")
        text_2_1_filepath = os.path.join(in_temp_dir, f"{file_id}.cwe")
        if os.path.exists(text_2_filepath):
            with open(text_2_filepath) as fin:
                cve = fin.read()
            if os.path.exists(text_2_1_filepath):
                with open(text_2_1_filepath) as fin:
                    cwe = fin.read()
            else:
                cwe = None
            new_instances.append({
                TEXT_1_COL: new_text,
                TEXT_2_COL: cve,
                TEXT_2_1_COL: cwe,
            })
        else:
            with open(out_java_filepath) as fin:
                new_instances.append({TEXT_COL: new_text})
    out_temp_dir.cleanup()
    return new_instances


def call_jt(in_temp_dir: str, seed: int, log_outfile: str = None):
    new_instances = []
    out_temp_dir = tempfile.TemporaryDirectory()
    jt_run_command = JT_COMMAND.format(jt_jar=JT_JAR_FILEPATH, in_dir=in_temp_dir, out_dir=out_temp_dir.name, seed=seed).split(" ")
    if log_outfile and os.path.exists(log_outfile):
        with open(log_outfile, "a") as fout:
            subprocess.call(jt_run_command, stdout=subprocess.DEVNULL, stderr=fout)
    else:
        subprocess.call(jt_run_command, stdout=subprocess.DEVNULL)

    for r in JT_RULES:
        rule_dirpath = os.path.join(out_temp_dir.name, r)
        if not os.path.exists(rule_dirpath):
            print_stdout_file(f"- JavaTransformer's rule \"{r}\" generated 0 new files", log_outfile)
            continue
        out_java_filepaths = [os.path.join(rule_dirpath, f) for f in os.listdir(rule_dirpath) if os.path.isfile(os.path.join(rule_dirpath, f)) and re.compile(r".*\.java").match(f)]
        print_stdout_file(f"- JavaTransformer's rule \"{r}\" generated {len(out_java_filepaths)} new files", log_outfile)
        for out_java_filepath in out_java_filepaths:
            with open(out_java_filepath) as fin:
                new_text = fin.read().strip()
            file_id = os.path.basename(out_java_filepath).split("_")[0]
            text_2_filepath = os.path.join(in_temp_dir, f"{file_id}.cve")
            text_2_1_filepath = os.path.join(in_temp_dir, f"{file_id}.cwe")
            if os.path.exists(text_2_filepath):
                with open(text_2_filepath) as fin:
                    cve = fin.read()
                if os.path.exists(text_2_1_filepath):
                    with open(text_2_1_filepath) as fin:
                        cwe = fin.read()
                else:
                    cwe = None
                new_instances.append({
                    TEXT_1_COL: new_text,
                    TEXT_2_COL: cve,
                    TEXT_2_1_COL: cwe,
                })
            else:
                with open(out_java_filepath) as fin:
                    new_instances.append({TEXT_COL: new_text})
    out_temp_dir.cleanup()
    return new_instances


def augment_minority_spat(input_ds: datasets.Dataset, times: int = 1, log_outfile: str = None):
    rt_jar_filepath = find_java_runtime_jar()
    if rt_jar_filepath is None:
        print_stdout_file("Could not find a JAR file containing the Java runtime. Data augmentation with SPAT cannot be run", log_outfile)
        return copy_dataset(input_ds)

    in_temp_dir = tempfile.TemporaryDirectory()
    input_df = pd.DataFrame(input_ds)
    minority_df = input_df[input_df[LABEL_COL] == 1]
    for idx, row in minority_df.iterrows():
        with open(os.path.join(in_temp_dir.name, f"{idx}.java"), "w") as fout:
            if TEXT_COL in row:
                java_text = row[TEXT_COL]
            elif TEXT_1_COL in row:
                java_text = row[TEXT_1_COL]
            fout.write(SPAT_PREFIX + java_text + SPAT_SUFFIX)
        if TEXT_2_COL in row:
            with open(os.path.join(in_temp_dir.name, f"{idx}.cve"), "w") as fout:
                fout.write(row[TEXT_2_COL])
        if TEXT_2_1_COL in row:
            if row[TEXT_2_1_COL]:
                with open(os.path.join(in_temp_dir.name, f"{idx}.cwe"), "w") as fout:
                    fout.write(row[TEXT_2_1_COL])
    all_new_instances = []
    # If run multiple times, we need to supply our repeatable sequence of seeds.
    if DETERMINISM:
        seed(RANDOM_SEED)
    for _ in range(0, times):
        spat_seed = get_random_long_seed()
        for rule in range(0, 18):
            all_new_instances.extend(call_spat(rule, in_temp_dir.name, rt_jar_filepath, spat_seed, log_outfile))
    in_temp_dir.cleanup()
    new_instances_df = pd.DataFrame(all_new_instances).drop_duplicates(ignore_index=True)
    new_instances_df[LABEL_COL] = 1
    print_stdout_file(f"SPAT generated {len(all_new_instances)} new instances in total, among which {len(new_instances_df)} are unique", log_outfile)
    new_ds = datasets.Dataset.from_pandas(pd.concat([input_df, new_instances_df], ignore_index=True))
    return new_ds


def augment_minority_jt(input_ds: datasets.Dataset, times: int = 1, log_outfile: str = None):
    in_temp_dir = tempfile.TemporaryDirectory()
    input_df = pd.DataFrame(input_ds)
    minority_df = input_df[input_df[LABEL_COL] == 1]
    for idx, row in minority_df.iterrows():
        with open(os.path.join(in_temp_dir.name, f"{idx}.java"), "w") as fout:
            if TEXT_COL in row:
                java_text = row[TEXT_COL]
            elif TEXT_1_COL in row:
                java_text = row[TEXT_1_COL]
            fout.write(java_text)
        if TEXT_2_COL in row:
            with open(os.path.join(in_temp_dir.name, f"{idx}.cve"), "w") as fout:
                fout.write(row[TEXT_2_COL])
        if TEXT_2_1_COL in row:
            if row[TEXT_2_1_COL]:
                with open(os.path.join(in_temp_dir.name, f"{idx}.cwe"), "w") as fout:
                    fout.write(row[TEXT_2_1_COL])
    all_new_instances = []
    # If run multiple times, we need to supply our repeatable sequence of seeds. The first run generates ~500 unique tests, while subsequent runs add ~100 unique tests each
    if DETERMINISM:
        seed(RANDOM_SEED)
    for _ in range(0, times):
        jt_seed = get_random_long_seed()
        all_new_instances.extend(call_jt(in_temp_dir.name, jt_seed, log_outfile))
    in_temp_dir.cleanup()
    new_instances_df = pd.DataFrame(all_new_instances).drop_duplicates(ignore_index=True)
    new_instances_df[LABEL_COL] = 1
    print_stdout_file(f"JavaTransformer generated {len(all_new_instances)} new instances in total, among which {len(new_instances_df)} are unique", log_outfile)
    new_ds = datasets.Dataset.from_pandas(pd.concat([input_df, new_instances_df], ignore_index=True))
    return new_ds


def augment_random_oversample_minority(input_ds: datasets.Dataset, ratio: float = 1.0, log_outfile: str = None):
    input_df = pd.DataFrame(input_ds)
    ros = RandomOverSampler(sampling_strategy=ratio, random_state=RANDOM_SEED)
    if TEXT_COL in input_df.columns:
        X_resampled, y_resampled = ros.fit_resample(input_df[TEXT_COL].to_frame(), input_df[LABEL_COL])
        new_ds = datasets.Dataset.from_pandas(pd.DataFrame({TEXT_COL: X_resampled[TEXT_COL], LABEL_COL: y_resampled}))
    elif TEXT_1_COL in input_df.columns and TEXT_2_COL in input_df.columns and TEXT_2_1_COL not in input_df.columns:
        X_resampled, y_resampled = ros.fit_resample(input_df[[TEXT_1_COL, TEXT_2_COL]], input_df[LABEL_COL])
        new_ds = datasets.Dataset.from_pandas(pd.DataFrame({TEXT_1_COL: X_resampled[TEXT_1_COL], TEXT_2_COL: X_resampled[TEXT_2_COL], LABEL_COL: y_resampled}))
    elif TEXT_1_COL in input_df.columns and TEXT_2_COL in input_df.columns and TEXT_2_1_COL in input_df.columns:
        X_resampled, y_resampled = ros.fit_resample(input_df[[TEXT_1_COL, TEXT_2_COL, TEXT_2_1_COL]], input_df[LABEL_COL])
        new_ds = datasets.Dataset.from_pandas(pd.DataFrame({TEXT_1_COL: X_resampled[TEXT_1_COL], TEXT_2_COL: X_resampled[TEXT_2_COL], TEXT_2_1_COL: X_resampled[TEXT_2_1_COL], LABEL_COL: y_resampled}))
    print_stdout_file(f"Random Oversampling generated {len(new_ds) - len(input_ds)} new instances", log_outfile)
    return new_ds


def augment_ds(input_ds: datasets.Dataset, tech: AugmentationApproach, extent: Union[float, int, None] = None, log_outfile: str = None):
    if tech == AugmentationApproach.NONE:
        print_stdout_file("No augmentation will be applied as requested.", log_outfile)
        return copy_dataset(input_ds)
    if AugmentationApproach.JT.value in tech:
        print_stdout_file(f"Applying data augmentation on the minority class of the training set using JavaTransformer ({extent} times)", log_outfile)
        return augment_minority_jt(input_ds, extent, log_outfile)
    if AugmentationApproach.SPAT.value in tech:
        print_stdout_file(f"Applying data augmentation on the minority class of the training set using SPAT ({extent} times)", log_outfile)
        return augment_minority_spat(input_ds, extent, log_outfile)
    if AugmentationApproach.BOOTSTRAP.value in tech:
        print_stdout_file(f"Applying random oversampling on the minority class of the training set until the two classes reaches {extent} ratio", log_outfile)
        return augment_random_oversample_minority(input_ds, extent, log_outfile)
    print_stdout_file("The requested augmentation approaches was not recognized. No further action taken.", log_outfile)
    return copy_dataset(input_ds)
