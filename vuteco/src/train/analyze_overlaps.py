import json
import os

import pandas as pd
from core.common.constants import (DATA_DIRPATH, EXPERIMENT_DIRPATH,
                                   EXPERIMENT_SESSION_DIRPATH, LABEL_COL,
                                   RANDOM_SEED, VUL4J_TEST_FILEPATH,
                                   End2EndName, FinderName, LinkerName)
from core.common.load_dataset import (load_vul4j_for_e2e, load_vul4j_for_fnd,
                                      load_vul4j_for_lnk)
from core.common.utils_training import split_dataset
from train.analyze_experiment import paired_overlap


def overlap_analysis(best_runs: dict, dataset: pd.DataFrame, tech_type: str):
    test_predictions = {}
    last_split_used = None
    for tech_name, best_run_file in best_runs.items():
        best_run_filepath = os.path.join(EXPERIMENT_SESSION_DIRPATH, tech_name, best_run_file)
        if not os.path.exists(best_run_filepath):
            continue
        with open(best_run_filepath) as fin:
            tech_best_res = json.load(fin)
        if last_split_used is None or last_split_used != tech_best_res["split"]:
            last_split_used = tech_best_res["split"]
            _, _, test_ds = split_dataset(dataset, ratios=last_split_used, label_col=LABEL_COL, seed=RANDOM_SEED)
            if str(last_split_used) not in test_predictions:
                test_predictions[str(last_split_used)] = {
                    "test_set": test_ds.to_pandas(),
                    "techniques": []
                }
        tech_test_res = tech_best_res["test_results"] if "test_results" in tech_best_res else tech_best_res
        if "test_predictions" in tech_test_res:
            test_predictions[str(last_split_used)]["techniques"].append((tech_name, tech_test_res["test_predictions"]))

    base_overlap_dirpath = os.path.join(EXPERIMENT_DIRPATH, "overlaps_of_best", tech_type)
    for _, split_data in test_predictions.items():
        test_df = split_data["test_set"]
        for i in range(len(split_data["techniques"])):
            for j in range(i + 1, len(split_data["techniques"])):
                tech_i_name, tech_i_preds = split_data["techniques"][i]
                tech_j_name, tech_j_preds = split_data["techniques"][j]
                if len(tech_i_preds) != len(tech_j_preds):
                    print(f"Mismatched length between {tech_i_name} and {tech_j_name}")
                    continue
                overlap_dirpath = os.path.join(base_overlap_dirpath, f"{tech_i_name}_{tech_j_name}")
                os.makedirs(overlap_dirpath, exist_ok=True)
                paired_overlap(tech_i_preds, tech_j_preds, tech_i_name, tech_j_name, test_df, overlap_dirpath)
    # TODO Do a master overlap again using all together (feasible?)


if __name__ == "__main__":
    best_fnd_runs = {
        FinderName.CODEBERT_FND.value: "2025_04_19_01_13_31/0.7-0.1-0.2_jt_wbce_False.json",
        FinderName.UNIXCODER_FND.value: "2025_04_19_01_13_32/0.7-0.1-0.2_none_wbce_True.json",
        FinderName.CODET5PLUS_FND.value: "2025_04_19_01_13_32/0.7-0.1-0.2_jt_bce_True.json",
        FinderName.CODELLAMA_FND.value: "2025_05_30_02_39_30/0.7-0.1-0.2_None_spat.json",
        FinderName.QWENCODER_FND.value: "2025_06_16_14_35_55/0.7-0.1-0.2_None_none.json",
        FinderName.DEEPSEEKCODER_FND.value: "2025_06_03_17_54_38/0.7-0.1-0.2_None_none.json",
        FinderName.FIX_FND.value: "2025_07_03_16_34_22/test_results.json",
    }
    best_e2e_runs = {
        End2EndName.CODEBERT_E2E.value: "2025_05_19_00_17_52/0.7-0.1-0.2_None_fl-meta_pt_none_jd_None_jt_wbce_none_bce_wbce_None_True.json",
        End2EndName.UNIXCODER_E2E.value: "2025_05_10_07_07_47/0.7-0.1-0.2_None_fl-meta_pt-ft_bs_jd_True_none_wbce_jt_wbce_bce_True_True.json",
        End2EndName.CODET5PLUS_E2E.value: "2025_05_19_00_20_00/0.7-0.1-0.2_None_fl-mask_pt_none_jd_None_jt_bce_jt_wbce_wbce_True_True.json",
        End2EndName.CODELLAMA_E2E.value: "2025_06_17_19_42_00/0.7-0.1-0.2_None_lo_ft_jt_True_none.json",
        End2EndName.QWENCODER_E2E.value: "2025_06_16_18_31_03/0.7-0.1-0.2_None_lo_ft_bs_None_none.json",
        End2EndName.DEEPSEEKCODER_E2E.value: "2025_06_23_10_19_46/0.7-0.1-0.2_None_lo_ft_bs_None_bs.json",
        End2EndName.FIX_E2E.value: "2025_07_03_16_36_40/test_results.json",
    }
    if len(best_fnd_runs) > 0:
        fnd_df = load_vul4j_for_fnd(VUL4J_TEST_FILEPATH, os.path.abspath(DATA_DIRPATH))
        overlap_analysis(best_fnd_runs, fnd_df, "fnd")
    if len(best_e2e_runs) > 0:
        e2e_df = load_vul4j_for_e2e(VUL4J_TEST_FILEPATH, os.path.abspath(DATA_DIRPATH))
        overlap_analysis(best_e2e_runs, e2e_df, "e2e")
