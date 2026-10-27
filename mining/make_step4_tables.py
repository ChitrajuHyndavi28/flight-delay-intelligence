"""Builds slide-ready markdown tables for Step 4 from the CSV outputs (no hand-typed numbers)
-> results/model_comparison/step4_tables.md"""
import pandas as pd, numpy as np
from common import RESULTS
MC, EX = RESULTS / "model_comparison", RESULTS / "explain"
A = pd.read_csv(MC / "model_comparison_t60_numeric.csv"); base = 0.1795
pm = lambda r, c, p=4: f"{r[c]:.{p}f}" + ("" if pd.isna(r[c + '_std']) else f" ± {r[c + '_std']:.{p}f}")
out = []
def md(df): return "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values)

# a. main table
rows = []
for _, r in A[(A.key != "dummy_majority") & (~A.key.str.contains("matched"))].iterrows():
    pt = f"{r.predict_time_s:.1f}" if r.key != "lightgbm" else "3.5†"
    rows.append({"Model": r.model, "Train rows": f"{int(r.n_train):,}", "PR-AUC (mean ± std over seeds)": pm(r, "pr_auc"), "Lift": f"{r.lift:.2f}x", "ROC-AUC": pm(r, "roc_auc"),
                 "F1": f"{r.f1:.3f}", "Precision": f"{r.precision:.3f}", "Recall": f"{r.recall:.3f}", "Train s": f"{r.train_time_s:.0f}", "Predict s": pt, "Seeds": int(r.n_seeds)})
r = A[A.key == "dummy_majority"].iloc[0]
rows.append({"Model": "Majority dummy", "Train rows": f"{int(r.n_train):,}", "PR-AUC (mean ± std over seeds)": f"{r.pr_auc:.4f}", "Lift": "1.00x", "ROC-AUC": "0.5000", "F1": "0.000", "Precision": "0.000", "Recall": "0.000", "Train s": "0", "Predict s": "0", "Seeds": 1})
out += [f"### (a) T-60 test set, one row per model (test base rate {base:.4f}; threshold = max-F1 on September)", md(pd.DataFrame(rows)),
        "† LightGBM model-only prediction time on 932,084 rows (3 runs); the cached training log also timed pandas slicing (62 s). Random Forest uses a 1M stratified sample, KNN a 200k stratified sample.", ""]
# b. matched
M = pd.read_csv(MC / "matched_training_data_comparison.csv").fillna(""); M = M.rename(columns={"pr_auc_boot_ci": "PR-AUC 95% CI (flight-level)", "pr_auc": "PR-AUC / diff", "lift": "Lift / verdict"})
out += ["### (b) Matched training data", md(M[["training_data", "model", "PR-AUC / diff", "Lift / verdict", "PR-AUC 95% CI (flight-level)", "roc_auc", "train_s"]]), ""]
# c. paired bootstrap
F = pd.read_csv(MC / "lightgbm_vs_xgboost_t60.csv"); C = pd.read_csv(MC / "cluster_bootstrap_by_day.csv"); C = C[C.scenario == "t60"].set_index("comparison")
rows = []
for _, r in F.iterrows():
    cc = C.loc[r.comparison] if r.comparison in C.index else None
    rows.append({"Comparison (T-60)": r.comparison, "Mean diff": f"{r.mean_diff_pr_auc:+.4f}", "95% CI, flight-level": f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]",
                 "95% CI, day-clustered": "" if cc is None else f"[{cc.day_cluster_ci_lo:+.4f}, {cc.day_cluster_ci_hi:+.4f}]", "Verdict (flight-level CI)": r.verdict})
out += ["### (c) LightGBM minus XGBoost, paired PR-AUC difference", md(pd.DataFrame(rows)), ""]
F2 = pd.read_csv(MC / "lightgbm_vs_xgboost_dayahead.csv"); C2 = pd.read_csv(MC / "cluster_bootstrap_by_day.csv"); C2 = C2[C2.scenario == "dayahead"].set_index("comparison")
rows = [{"Comparison (day-ahead)": r.comparison, "Mean diff": f"{r.mean_diff_pr_auc:+.4f}", "95% CI, flight-level": f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]",
         "95% CI, day-clustered": "" if r.comparison not in C2.index else f"[{C2.loc[r.comparison].day_cluster_ci_lo:+.4f}, {C2.loc[r.comparison].day_cluster_ci_hi:+.4f}]"} for _, r in F2.iterrows()]
out += [md(pd.DataFrame(rows)), ""]
# d. ablation
Ab = pd.read_csv(MC / "ablation_lightgbm.csv"); E = pd.read_csv(MC / "ablation_effects_paired.csv")
out += ["### (d) LightGBM ablation (test; CIs flight-level bootstrap on seed 42)", md(Ab[["setting", "n_features", "seeds", "pr_auc", "pr_auc_ci", "roc_auc", "roc_auc_ci"]]), "", "Paired effects:", md(E), ""]
# e. SHAP + rules
S = pd.read_csv(EX / "shap_mean_abs.csv").head(10); G = pd.read_csv(EX / "lgbm_gain_by_group.csv")
out += ["### (e) Top 10 SHAP features (mean |SHAP|, log-odds, 50k test flights)", md(S.assign(mean_abs_shap=S.mean_abs_shap.round(3))), "", "Gain by feature group:", md(G), ""]
# f. threshold transfer
T = A[(A.key != "dummy_majority") & (~A.key.str.contains("matched"))][["model", "threshold", "val_f1", "f1", "oracle_f1", "predicted_positive_rate"]].copy(); T["loss"] = T.oracle_f1 - T.f1
T = T.rename(columns={"threshold": "Sep-tuned threshold", "val_f1": "Val F1 (Sep, at its best thr)", "f1": "Test F1 @ Sep threshold", "oracle_f1": "Best possible test F1 (diagnostic)", "predicted_positive_rate": "Test predicted-positive rate", "loss": "F1 lost to transfer"})
for c in T.columns[1:]: T[c] = T[c].map("{:.4f}".format)
out += [f"### (f) Threshold transfer (val prevalence 13.0% -> test prevalence {100*base:.2f}%)", md(T), ""]
ct = A[(A.key != "dummy_majority") & (~A.key.str.contains("matched"))].copy()
cmp_ = pd.DataFrame({"model": ct.model, "train_rows": ct.n_train.astype(int), "pr_auc_mean": ct.pr_auc.round(4), "pr_auc_std": ct.pr_auc_std.round(4), "lift_over_base_rate": ct.lift.round(2),
                     "roc_auc": ct.roc_auc.round(4), "f1": ct.f1.round(4), "precision": ct.precision.round(4), "recall": ct.recall.round(4),
                     "train_s": ct.train_time_s.round(1), "predict_s": ct.predict_time_s.round(1), "seeds": ct.n_seeds})
lat = pd.read_csv(EX / "lightgbm_predict_latency.csv").predict_seconds_median.iloc[0]
cmp_["predict_note"] = ""; m_ = cmp_.model.eq("LightGBM (main)"); cmp_.loc[m_, "predict_s"] = round(float(lat), 1)
cmp_.loc[m_, "predict_note"] = "model-only time on 932,084 test rows (training log also timed pandas slicing: 62 s)"
cmp_.to_csv(MC / "comparison_table.csv", index=False)
(MC / "step4_tables.md").write_text("\n".join(out)); print("\n".join(out)[:200]); print("written", MC / "step4_tables.md")
