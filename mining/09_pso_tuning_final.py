"""Step 4e: particle-swarm tuning (as in the base paper) of XGBoost and LightGBM for both prediction moments (T-60, day-ahead),
then a full-data refit with weather ON and OFF to quantify the weather effect with the tuned models.
Tuning uses a stratified train sample and VALIDATION PR-AUC only; the test months are touched once, for the final refit."""
import json, time, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from common import ROOT, SEED
from ml_common import (CACHE, CAT_COLS, LGB_BASE, SCENARIOS, TARGET, PoissonBootstrapAP, best_f1_threshold, class_weights, load_features,
                       make_xgb, metrics_at, ordinal_codes, split_masks, stratified_idx, assert_no_leakage)
from pso import pso

OUT = ROOT / "results" / "model_comparison"; PRED = CACHE / "preds"
N_TUNE = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
df = load_features(); tr, va, te = split_masks(df); y = df[TARGET].to_numpy(np.int8); yva, yte = y[va], y[te]
CW = class_weights(y[tr]); W = np.where(y == 1, CW[1], CW[0]).astype("float32")
assert_no_leakage(SCENARIOS["t60"])
XGB_SPACE = [("max_depth", 3, 12, "int"), ("learning_rate", 0.05, 0.25, "log"), ("min_child_weight", 1, 200, "log"),
             ("colsample_bytree", 0.4, 1.0, "lin"), ("gamma", 0.0, 2.0, "lin"), ("subsample", 0.5, 1.0, "lin")]
LGB_SPACE = [("num_leaves", 15, 511, "logint"), ("learning_rate", 0.05, 0.25, "log"), ("min_child_samples", 20, 2000, "logint"),
             ("feature_fraction", 0.4, 1.0, "lin"), ("lambda_l2", 0.01, 30.0, "log"), ("bagging_fraction", 0.5, 1.0, "lin")]


def fit_xgb(feats, p, idx, seed):
    X = ordinal_codes(df, feats); p = dict(p)
    m = make_xgb({k: v for k, v in p.items() if k != "subsample"}, seed); m.set_params(subsample=p["subsample"])
    m.fit(X.iloc[idx], y[idx], sample_weight=W[idx], eval_set=[(X[va], yva)], sample_weight_eval_set=[W[va]], verbose=False)
    return m, X


def fit_lgb(feats, p, idx, seed):
    cat = [c for c in feats if c in CAT_COLS]; Xd = df[feats]
    dtr = lgb.Dataset(Xd.iloc[idx], y[idx], weight=W[idx], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(Xd[va], yva, categorical_feature=cat, reference=dtr)
    b = lgb.train({**LGB_BASE, **p, "seed": seed, "num_threads": 10}, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    return b, Xd


def predict(kind, model, X):
    if kind == "xgb": return model.predict_proba(X[va])[:, 1], model.predict_proba(X[te])[:, 1]
    return model.predict(X[va], num_iteration=model.best_iteration), model.predict(X[te], num_iteration=model.best_iteration)


best = json.load(open(OUT / "pso_best_params.json")) if (OUT / "pso_best_params.json").exists() else {}
for scn in ["t60", "dayahead"]:
    feats = SCENARIOS[scn]; idx = stratified_idx(y, tr, N_TUNE, SEED)
    for kind, space, fit in [("xgb", XGB_SPACE, fit_xgb), ("lgb", LGB_SPACE, fit_lgb)]:
        key = f"{scn}/{kind}"
        if key in best: print("cached", key, best[key]); continue
        print(f"=== PSO {key} on {len(idx):,} train rows ===", flush=True); t0 = time.time()
        def obj(p):
            m, X = fit(feats, p, idx, SEED); sv, _ = predict(kind, m, X); return average_precision_score(yva, sv)
        p, s, hist = pso(obj, space, seed=SEED, log_path=OUT / f"pso_history_{scn}_{kind}.json")
        best[key] = dict(params=p, val_pr_auc_sample=s, minutes=round((time.time() - t0) / 60, 1)); json.dump(best, open(OUT / "pso_best_params.json", "w"), indent=1)
        print("BEST", key, best[key], flush=True)

# ---- final refits on the FULL training set: weather ON vs OFF, tuned params ----------------------------------------
full = np.nonzero(tr)[0]; rows = []; scores = {}
for scn_label, scn_on, scn_off in [("T-60", "t60", "t60_no_weather"), ("Day-ahead", "dayahead", "dayahead_no_weather")]:
    for kind, fit, name in [("xgb", fit_xgb, "XGBoost (PSO-tuned)"), ("lgb", fit_lgb, "LightGBM (PSO-tuned)")]:
        p = best[f"{scn_on}/{kind}"]["params"]
        for wx, scn in [("ON", scn_on), ("OFF", scn_off)]:
            t0 = time.time(); m, X = fit(SCENARIOS[scn], p, full, SEED); sv, st = predict(kind, m, X); tt = time.time() - t0
            np.savez(PRED / scn / f"{kind}_pso__{SEED}.npz" if (PRED / scn).exists() else (PRED / f"{scn}_{kind}_pso.npz"), val=sv.astype("float32"), test=st.astype("float32"))
            thr, _ = best_f1_threshold(yva, sv); mt = metrics_at(yte, st, thr); scores[(scn_label, name, wx)] = st
            rows.append(dict(task=scn_label, model=name, weather=wx, n_features=len(SCENARIOS[scn]), pr_auc=mt["pr_auc"], roc_auc=mt["roc_auc"], f1=mt["f1"],
                             precision=mt["precision"], recall=mt["recall"], accuracy=mt["accuracy"], train_s=round(tt, 1)))
            print(rows[-1], flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT / "tuned_final_weather_on_off.csv", index=False)

# paired bootstrap of the weather effect (ON minus OFF) with the tuned models
B = PoissonBootstrapAP(yte, n_boot=300); eff = []
for task in ["T-60", "Day-ahead"]:
    for name in ["XGBoost (PSO-tuned)", "LightGBM (PSO-tuned)"]:
        r = B.run({"on": scores[(task, name, "ON")], "off": scores[(task, name, "OFF")]}, metrics=("ap", "auc"))
        d_ap, d_auc = r["ap"]["on"] - r["ap"]["off"], r["auc"]["on"] - r["auc"]["off"]
        a = res[(res.task == task) & (res.model == name)].set_index("weather")
        eff.append(dict(task=task, model=name, pr_auc_off=a.loc["OFF", "pr_auc"], pr_auc_on=a.loc["ON", "pr_auc"], delta_pr_auc=a.loc["ON", "pr_auc"] - a.loc["OFF", "pr_auc"],
                        delta_pr_auc_ci=f"[{np.percentile(d_ap, 2.5):.4f}, {np.percentile(d_ap, 97.5):.4f}]", delta_roc_auc=a.loc["ON", "roc_auc"] - a.loc["OFF", "roc_auc"],
                        delta_roc_auc_ci=f"[{np.percentile(d_auc, 2.5):.4f}, {np.percentile(d_auc, 97.5):.4f}]", delta_recall=a.loc["ON", "recall"] - a.loc["OFF", "recall"]))
pd.DataFrame(eff).to_csv(OUT / "tuned_weather_effect.csv", index=False); print(pd.DataFrame(eff).round(4).to_string(index=False))
