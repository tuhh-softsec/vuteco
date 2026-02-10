import json
import os
import tempfile
from random import random, seed

import numpy as np
import pandas as pd
from core.common.constants import (ANALYSIS_PERFORMANCE_FILENAME, DATA_DIRPATH,
                                   EXPERIMENT_ANALYSIS_DIRPATH,
                                   EXPERIMENT_SESSION_DIRPATH, LABEL_COL,
                                   PR_CURVES_DIRNAME, RANDOM_SEED,
                                   ROC_CURVES_DIRNAME, TEXT_1_COL, TEXT_2_COL,
                                   VUL4J_TEST_FILEPATH, CommonConfigKeys,
                                   End2EndName, FinderName, LinkerName)
from core.common.load_dataset import (load_vul4j_for_e2e, load_vul4j_for_fnd,
                                      load_vul4j_for_lnk)
from core.common.utils_training import (compute_grouped_estimations,
                                        compute_performance_clf, split_dataset)
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from sklearn import metrics
from tqdm import tqdm


def threshold_analysis(curve, auc_val, curve_fun, axis_labels, out_path, out_filename, test_ds, probas):
    x, y, thresholds = curve_fun(test_ds[LABEL_COL], probas)
    plt.plot(x, y, label=f"AUC-{curve}={str(auc_val)}")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='dashed', color="black")
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.legend(loc=4)
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(out_path, out_filename + ".pdf"), bbox_inches='tight')
    plt.close()

    # ROC Threshold tuning
    scores = [compute_performance_clf(test_ds[LABEL_COL], np.where(np.array(probas) >= t, 1, 0)) for t in tqdm(thresholds)]
    f1_optimal_thres = thresholds[max(range(len(scores)), key=lambda x: scores[x]['f1'] if scores[x]['f1'] is not None else float('-inf'))]
    f05_optimal_thres = thresholds[max(range(len(scores)), key=lambda x: scores[x]['f05'] if scores[x]['f05'] is not None else float('-inf'))]
    # j_optimal_thres = thresholds[np.argmax(y - x)]
    # e_optimal_thres = thresholds[np.argmin(np.sqrt((1 - y)**2 + x**2))]
    # normal_scores = compute_performance_clf(test_ds[LABEL_COL], np.where(np.array(probas) >= 0.5, 1, 0))
    f1_scores = compute_performance_clf(test_ds[LABEL_COL], np.where(np.array(probas) >= f1_optimal_thres, 1, 0))
    f05_scores = compute_performance_clf(test_ds[LABEL_COL], np.where(np.array(probas) >= f05_optimal_thres, 1, 0))
    # j_scores = compute_performance_clf(test_ds[LABEL_COL], np.where(np.array(probas) >= j_optimal_thres, 1, 0))
    # e_scores = compute_performance_clf(test_ds[LABEL_COL], np.where(np.array(probas) >= e_optimal_thres, 1, 0))
    # print(f"Normal Threshold: 0.5")
    # print(normal_scores)
    # print(f"F1-based Threshold: {f1_optimal_thres}")
    # print(f1_scores)
    # print(f"F0.5-based Threshold: {f05_optimal_thres}")
    # print(f05_scores)
    # print(f"Youden's J-based Threshold: {j_optimal_thres}")
    # print(j_scores)
    # print(f"Euclidean-based Threshold: {e_optimal_thres}")
    # print(e_scores)
    # print()
    f1_opt_perf = {f"OptF1_{curve}_{k}": v for k, v in f1_scores.items()}
    f05_opt_perf = {f"OptF05_{curve}_{k}": v for k, v in f05_scores.items()}
    extra = {
        f"OptF1_{curve}_thresh": f1_optimal_thres,
        **f1_opt_perf,
        f"OptF05_{curve}_thresh": f05_optimal_thres,
        **f05_opt_perf
    }
    return extra


def measure_paired_agreement(pred_1: list[int], pred_2: list[int], name_1: str, name_2: str, dest_dir: str, positive_indexes: list[int] = None):
    if positive_indexes is not None:
        indexes = positive_indexes
        left = [pred_1[i] for i in positive_indexes]
        right = [pred_2[i] for i in positive_indexes]
    else:
        left = pred_1
        right = pred_2
        indexes = list(range(len(pred_1)))
    c_kappa = metrics.cohen_kappa_score(left, right)
    and_result = [x & y for x, y in zip(left, right)]
    or_result = [x | y for x, y in zip(left, right)]
    xor_result = [x ^ y for x, y in zip(left, right)]
    and_ratio = sum(and_result) / len(and_result)
    or_ratio = sum(or_result) / len(or_result)
    jaccard = float(sum(and_result)) / sum(or_result) if sum(or_result) > 0 else None
    disagreement = sum(xor_result) / len(xor_result)
    agreement = 1.0 - disagreement
    venn2(subsets=(
        set(i for i in indexes if pred_1[i] == 1),
        set(i for i in indexes if pred_2[i] == 1)),
        set_labels=(name_1, name_2)
    )
    key_prefix = "positive" if positive_indexes is not None else "total"
    plt.savefig(os.path.join(dest_dir, f"{key_prefix}_overlap.pdf"), format="pdf")
    plt.close()
    return {
        f"{key_prefix}_cohen_kappa": c_kappa,
        f"{key_prefix}_and_ratio": and_ratio,
        f"{key_prefix}_or_ratio": or_ratio,
        f"{key_prefix}_jaccard": jaccard,
        f"{key_prefix}_disagreement": disagreement,
        f"{key_prefix}_agreement": agreement,
    }


def baseline_performance(test_ds):
    str_ratio = float(len(test_ds.filter(lambda x: x[LABEL_COL] == 0))) / len(test_ds[LABEL_COL])
    with tempfile.TemporaryDirectory() as tmp_dirname:
        opt_scores = compute_performance_clf(test_ds[LABEL_COL], [1] * len(test_ds[LABEL_COL]))
        pes_scores = compute_performance_clf(test_ds[LABEL_COL], [0] * len(test_ds[LABEL_COL]))
        seed(RANDOM_SEED)
        rnd_scores = compute_performance_clf(test_ds[LABEL_COL], [1 if random() >= 0.5 else 0] * len(test_ds[LABEL_COL]))
        seed(RANDOM_SEED)
        str_scores = compute_performance_clf(test_ds[LABEL_COL], [1 if random() >= str_ratio else 0] * len(test_ds[LABEL_COL]))
    baseline_perf = {
        "opt": opt_scores,
        "pes": pes_scores,
        "rnd": rnd_scores,
        "str": str_scores
    }
    return pd.DataFrame.from_dict(baseline_perf, orient="index").reset_index()


def export_predictions(test_df, tech_out_dirpath: str, run_id: str):
    pred_dirpath = os.path.join(tech_out_dirpath, "predictions", run_id)
    os.makedirs(pred_dirpath, exist_ok=True)
    test_df[((test_df["PRED"] == 1) & (test_df[LABEL_COL] == 1))].to_csv(os.path.join(pred_dirpath, "tp.csv"), index=False)
    test_df[((test_df["PRED"] == 1) & (test_df[LABEL_COL] == 0))].to_csv(os.path.join(pred_dirpath, "fp.csv"), index=False)
    test_df[((test_df["PRED"] == 0) & (test_df[LABEL_COL] == 1))].to_csv(os.path.join(pred_dirpath, "fn.csv"), index=False)


def performance_metric_analysis(tech_results, tech_out_dirpath: str, perf_metrics):
    tech_result_df = pd.DataFrame(tech_results)
    if len(tech_result_df) == 0:
        return
    tech_result_df.drop_duplicates([c for c in tech_result_df.columns if c not in perf_metrics], inplace=True, keep="first")
    os.makedirs(tech_out_dirpath, exist_ok=True)
    tech_result_df.to_csv(os.path.join(tech_out_dirpath, ANALYSIS_PERFORMANCE_FILENAME), index=False)
    for metr in perf_metrics:
        nona_app_result_df = tech_result_df.copy()
        nona_app_result_df[metr].replace([np.inf, -np.inf], np.nan).dropna(inplace=True)
        aggregates = []
        aggregates.append({
            "config_name": "",
            "config_value": "",
            "count": len(tech_result_df[metr]),
            "numCount": len(nona_app_result_df[metr]),
            "mean": nona_app_result_df[metr].mean() if len(nona_app_result_df[metr].dropna()) > 1 else None,
            "min": nona_app_result_df[metr].min() if len(nona_app_result_df[metr].dropna()) > 1 else None,
            "25%": nona_app_result_df[metr].quantile(q=.25) if len(nona_app_result_df[metr].dropna()) > 1 else None,
            "med": nona_app_result_df[metr].median() if len(nona_app_result_df[metr].dropna()) > 1 else None,
            "75%": nona_app_result_df[metr].quantile(q=.75) if len(nona_app_result_df[metr].dropna()) > 1 else None,
            "max": nona_app_result_df[metr].max() if len(nona_app_result_df[metr].dropna()) > 1 else None
        })
        """
        for col in [c for c in nona_app_result_df.columns if c not in perf_metrics]:
            for group_n, group_df in nona_app_result_df.groupby(col):
                aggregates.append({
                    "config_name": col,
                    "config_value": group_n,
                    "count": len(group_df[metr]),
                    "mean": group_df[metr].mean() if len(group_df[metr].dropna()) > 1 else None,
                    "min": group_df[metr].min() if len(group_df[metr].dropna()) > 1 else None,
                    "25%": group_df[metr].quantile(q=.25) if len(group_df[metr].dropna()) > 1 else None,
                    "med": group_df[metr].median() if len(group_df[metr].dropna()) > 1 else None,
                    "75%": group_df[metr].quantile(q=.75) if len(group_df[metr].dropna()) > 1 else None,
                    "max": group_df[metr].max() if len(group_df[metr].dropna()) > 1 else None
                })
        """
        pd.DataFrame(aggregates).to_csv(os.path.join(tech_out_dirpath, f"agg_{metr}.csv"), index=False)


def paired_overlap(pred_1: list, pred_2: list, name_1: str, name_2: str, test_df: pd.DataFrame, overlap_dirpath: str):
    total_agreement_scores = measure_paired_agreement(pred_1, pred_2, name_1, name_2, overlap_dirpath)
    pos_indexes_test_set = test_df.index[test_df[LABEL_COL] == 1].tolist()
    pos_agreement_scores = measure_paired_agreement(pred_1, pred_2, name_1, name_2, overlap_dirpath, pos_indexes_test_set)
    pos_pred_1 = [pred_1[i] for i in pos_indexes_test_set]
    pos_pred_2 = [pred_2[i] for i in pos_indexes_test_set]
    with open(os.path.join(overlap_dirpath, f"{name_1}_agreement.json"), 'w') as fout:
        json.dump(total_agreement_scores | pos_agreement_scores, fout, indent=2)
    # pos_fix_test_res_indexes = [i for i in positive_indexes if fix_test_res_pred[i] == 1]
    # pos_test_res_indexes = [i for i in positive_indexes if test_res_pred[i] == 1]
    predicted_by = []
    for one, two in zip(pos_pred_1, pos_pred_2):
        if one == 1 and two == 0:
            predicted_by.append(name_1)
        elif one == 0 and two == 1:
            predicted_by.append(name_2)
        elif one == 1 and two == 1:
            predicted_by.append("Both")
        else:
            predicted_by.append("None")
    pos_df = test_df[test_df[LABEL_COL] == 1].copy()
    pos_df["FROM"] = predicted_by
    pos_df.to_csv(os.path.join(overlap_dirpath, "positives.csv"))


def analyze_techniques(runs_to_analyze: dict,
                       perf_metrics: list,
                       dataset: pd.DataFrame,
                       aliases: dict[str, str] = {},
                       fix_run: str = None):
    if fix_run is not None:
        fix_run_result_file = os.path.join(EXPERIMENT_SESSION_DIRPATH, fix_run, "test_results.json")
        with open(fix_run_result_file) as fin:
            fix_res = json.load(fin)
    else:
        fix_res = None
    # Load results from each approach, and analyze individually
    all_results = []
    train_times = {}
    for tech_name, run_dirs in runs_to_analyze.items():
        last_split_used = None
        tech_out_dirpath = os.path.join(EXPERIMENT_ANALYSIS_DIRPATH, tech_name)
        os.makedirs(tech_out_dirpath, exist_ok=True)
        tech_results = []
        for a_run_dir in run_dirs:
            tech_results_dirpath = os.path.join(EXPERIMENT_SESSION_DIRPATH, tech_name, a_run_dir)
            if not os.path.exists(tech_results_dirpath):
                continue
            res_files = [os.path.abspath(f.path) for f in os.scandir(tech_results_dirpath) if f.is_file() and ".json" in f.path]
            for a_res_file in res_files:
                with open(a_res_file) as fin:
                    tech_run_res = json.load(fin)
                if "train_time" in tech_run_res:
                    if tech_name not in train_times:
                        train_times[tech_name] = []
                    train_seconds = pd.to_timedelta(tech_run_res["train_time"]["total"] if isinstance(tech_run_res["train_time"], dict) else tech_run_res["train_time"]).total_seconds()
                    if "hyperparam_results" in tech_run_res:
                        for hp in tech_run_res["hyperparam_results"]:
                            train_seconds += pd.to_timedelta(hp["train_time"]).total_seconds()
                    train_times[tech_name].append(train_seconds)
                # Possible remapping of keys, if needed
                for old, new in aliases.items():
                    if "model_config" in tech_run_res and old in tech_run_res["model_config"]:
                        tech_run_res["model_config"][new] = tech_run_res["model_config"].pop(old)
                    if "hyperparams" in tech_run_res and old in tech_run_res["hyperparams"]:
                        tech_run_res["hyperparams"][new] = tech_run_res["hyperparams"].pop(old)
                # Load its test set
                if last_split_used is None or last_split_used != tech_run_res["split"]:
                    last_split_used = tech_run_res["split"]
                    _, _, test_ds = split_dataset(dataset, ratios=last_split_used, label_col=LABEL_COL, seed=RANDOM_SEED)
                    if TEXT_1_COL in test_ds.column_names:
                        print(f"Unique test cases in test set: {len(set(test_ds[TEXT_1_COL]))}")
                    if TEXT_2_COL in test_ds.column_names:
                        print(f"Unique vuln descriptions in test set: {len(set(test_ds[TEXT_2_COL]))}")
                    # Four baseline classifiers for this test set
                    baseline_df = baseline_performance(test_ds)
                    baseline_df.to_csv(
                        os.path.join(tech_out_dirpath, f"{'-'.join(str(s) for s in last_split_used)}_baseline.csv"),
                        index=False
                    )
                tech_test_res = tech_run_res["test_results"] if "test_results" in tech_run_res else tech_run_res
                tech_test_perf = {k.removeprefix("test_"): v for k, v in tech_test_res["test_performance"].items()}
                run_config = os.path.basename(a_res_file.removesuffix(".json"))
                extra_stuff = {}
                if "test_predictions" in tech_test_res:
                    test_df = test_ds.to_pandas()
                    test_df["PRED"] = tech_test_res["test_predictions"]
                    if "test_probabilities" in tech_test_res:
                        test_df["PROBA"] = tech_test_res["test_probabilities"]
                    export_predictions(test_df, tech_out_dirpath, run_config)
                    # train_df = train_ds.to_pandas()
                    # train_df[(train_df[LABEL_COL] == 1)].to_csv(os.path.join(pred_dirpath, "train_pos.csv"), index=False)

                    # Only for the Matching task
                    if TEXT_1_COL in test_df.columns and TEXT_2_COL in test_df.columns:
                        tc_wise_test_perf = compute_grouped_estimations(test_df, "PRED", LABEL_COL, TEXT_1_COL, TEXT_2_COL, "test_case")
                        cve_wise_test_perf = compute_grouped_estimations(test_df, "PRED", LABEL_COL, TEXT_2_COL, TEXT_1_COL, "vuln_descr")
                        extra_stuff.update(tc_wise_test_perf)
                        extra_stuff.update(cve_wise_test_perf)

                    # Agreement/overlap analysis with FixCommit
                    if fix_res is not None and fix_res["split"] == tech_run_res["split"] and fix_res["dataset_seed"] == tech_run_res["dataset_seed"]:
                        overlap_dirpath = os.path.join(tech_out_dirpath, "overlaps", run_config)
                        os.makedirs(overlap_dirpath, exist_ok=True)
                        paired_overlap(fix_res["test_results"]["test_predictions"], tech_test_res["test_predictions"], "fix", tech_name, test_df, overlap_dirpath)
                
                if "test_responses" in tech_test_res:
                    responses = tech_test_res["test_responses"]
                    invalid_responses = sum(1 if "0" not in r and "1" not in r else 0 for r in responses)
                    if invalid_responses > 0:
                        print(f"[{tech_name}-{run_config}] Nr. responses without 0/1: {invalid_responses} (out of {len(responses)})")

                # ROC and PR Curves
                if "test_probabilities" in tech_test_res:
                    probas = tech_test_res["test_probabilities"]
                    if any(x != 1.0 for x in probas):
                        out_filename = os.path.basename(os.path.normpath(a_res_file)).removeprefix(".json")
                        roc_extra = threshold_analysis("ROC",
                                                       tech_test_perf['auc_roc'], metrics.roc_curve, ["FPR", "TPR"],
                                                       os.path.join(tech_out_dirpath, ROC_CURVES_DIRNAME),
                                                       out_filename, test_ds, probas)
                        pr_extra = threshold_analysis("PR",
                                                      tech_test_perf['auc_pr'], metrics.precision_recall_curve, ["Precision", "Recall"],
                                                      os.path.join(tech_out_dirpath, PR_CURVES_DIRNAME),
                                                      out_filename, test_ds, probas)
                        extra_stuff.update(roc_extra)
                        extra_stuff.update(pr_extra)
                tech_results.append({
                    CommonConfigKeys.SPLIT.value: str(tech_run_res[CommonConfigKeys.SPLIT]),
                    **tech_run_res["model_config"],
                    **(tech_run_res["hyperparams"] if "hyperparams" in tech_run_res else {}),
                    **{k: v for k, v in tech_test_perf.items() if k in perf_metrics},
                    **extra_stuff,
                })
                all_results.append({
                    "datetime": a_run_dir,
                    "approach": tech_name,
                    "run": run_config,
                    "id": a_run_dir + "_" + tech_name + "_" + run_config,
                    CommonConfigKeys.SPLIT.value: str(tech_run_res[CommonConfigKeys.SPLIT]),
                    **tech_test_perf,
                    **extra_stuff,
                    "predictions": tech_test_res["test_predictions"],
                    "scores": tech_test_res.get(next((k for k in ["test_probabilities", "test_scores", "test_matches"] if k in tech_test_res), None), None)
                })
        time_stats = {
            k: {
                "number": len(times),
                "total": sum(times),
                "average": sum(times) / len(times) if times else 0.0
            } for k, times in train_times.items()
        }
        all_times = [v for values in train_times.values() for v in values]
        grand_total = sum(all_times)
        grand_average = grand_total / len(all_times) if all_times else 0.0
        time_stats["number"] = len(all_times)
        time_stats["total"] = grand_total
        time_stats["average"] = grand_average
        # Performance metric analysis
        performance_metric_analysis(tech_results, tech_out_dirpath, perf_metrics)
    return pd.DataFrame(all_results), time_stats


if __name__ == "__main__":
    fnd_runs = {
        FinderName.CODEBERT_FND.value: ["2025_04_19_01_13_31"],
        FinderName.UNIXCODER_FND.value: ["2025_04_19_01_13_32"],
        FinderName.CODET5PLUS_FND.value: ["2025_04_19_01_13_32"],
        FinderName.CODELLAMA_FND.value: ["2025_05_30_02_39_30"],
        FinderName.QWENCODER_FND.value: ["2025_06_16_14_35_55"],
        FinderName.DEEPSEEKCODER_FND.value: ["2025_06_03_17_54_38"],
        FinderName.GREP_FND.value: ["2025_07_04_17_31_14"],
        FinderName.VOCABULARY_FND.value: ["2025_09_17_11_54_16", "2025_09_17_12_02_33"]
    }
    e2e_runs = {
        End2EndName.CODEBERT_E2E.value: ["2025_05_19_00_17_52", "2025_08_08_10_22_47", "2025_09_10_19_16_13"],
        End2EndName.UNIXCODER_E2E.value: ["2025_05_10_07_07_47", "2025_08_08_10_22_47"],
        End2EndName.CODET5PLUS_E2E.value: ["2025_05_19_00_20_00", "2025_08_08_10_22_32"],
        End2EndName.CODELLAMA_E2E.value: ["2025_08_10_08_39_37", "2025_08_15_06_15_01", "2025_08_27_12_31_20"],
        End2EndName.QWENCODER_E2E.value: ["2025_08_03_08_39_07", "2025_08_13_06_20_19", "2025_09_01_23_54_27"],
        End2EndName.DEEPSEEKCODER_E2E.value: ["2025_08_13_04_39_09", "2025_08_17_08_39_19", "2025_09_01_17_00_13"],
        End2EndName.GREP_E2E.value: ["2025_07_11_18_02_15"],
        End2EndName.FIX_E2E.value: ["2025_07_03_16_36_40"],
        End2EndName.SIM_E2E.value: ["2025_09_16_13_46_55", "2025_09_16_17_01_01", "2025_09_16_20_59_53", "2025_09_17_01_10_43"],
    }
    lnk_runs = {
        LinkerName.CODEBERT_LNK.value: ["2025_05_06_18_01_59"],
        LinkerName.UNIXCODER_LNK.value: ["2025_05_03_04_47_51"],
        LinkerName.CODET5PLUS_LNK.value: ["2025_05_03_06_05_12"],
        LinkerName.CODELLAMA_LNK.value: ["2025_06_05_12_14_00"],
        LinkerName.QWENCODER_LNK.value: ["2025_06_11_04_17_39"],
        LinkerName.DEEPSEEKCODER_LNK.value: ["2025_06_16_09_55_55"],
    }
    perf_metrics = ["p_actu", "n_actu", "p_pred", "n_pred", "tp", "tn", "fp", "fn", "pre", "rec", "spe", "fpr", "fnr", "acc", "f1", "mcc", "f05", "auc_roc", "ir"]

    if len(fnd_runs) > 0:
        fnd_df = load_vul4j_for_fnd(VUL4J_TEST_FILEPATH, os.path.abspath(DATA_DIRPATH))
        print("Analyzing the Finder models")
        fnd_res_df, fnd_train_times = analyze_techniques(fnd_runs, perf_metrics, fnd_df, aliases={"normalize": "one_line"})
        with open(os.path.join(EXPERIMENT_ANALYSIS_DIRPATH, "fnd_train_times.json"), "w") as f:
            json.dump(fnd_train_times, f, indent=2)
    print("FND done!")

    if len(e2e_runs) > 0:
        e2e_ds = load_vul4j_for_e2e(VUL4J_TEST_FILEPATH, os.path.abspath(DATA_DIRPATH))
        print("Analyzing the End-to-End models")
        e2e_res_df, e2e_train_times = analyze_techniques(e2e_runs, perf_metrics, e2e_ds, fix_run="fix-e2e/2025_07_03_16_36_40")
        with open(os.path.join(EXPERIMENT_ANALYSIS_DIRPATH, "e2e_train_times.json"), "w") as f:
            json.dump(e2e_train_times, f, indent=2)
    print("E2E done!")

    if len(lnk_runs) > 0:
        lnk_ds = load_vul4j_for_lnk(VUL4J_TEST_FILEPATH, os.path.abspath(DATA_DIRPATH))
        print("Analyzing the Linker models")
        lnk_res_df, lnk_train_times = analyze_techniques(lnk_runs, perf_metrics, lnk_ds)
        with open(os.path.join(EXPERIMENT_ANALYSIS_DIRPATH, "lnk_train_times.json"), "w") as f:
            json.dump(lnk_train_times, f, indent=2)
    print("LNK done!")

    print("All done!")
