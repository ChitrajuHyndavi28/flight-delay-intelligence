"""Step 4d: 'post-departure' scenario = T-60 features + this flight's own departure delay (the paper's feature set (1)+(3)).
Departure delay is only known after take-off, so this is a comparison scenario, NOT an operational model. It deliberately
bypasses the T-60 leakage whitelist; nothing else in the project imports it. Read-only on the database."""
import json, time
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import average_precision_score
from common import ROOT, SEED, connect
from ml_common import (CAT_COLS, LGB_BASE, NUM_COLS, SCENARIOS, TARGET, XGB_DEFAULT, best_f1_threshold, class_weights, load_features,
                       make_xgb, metrics_at, ordinal_codes, split_masks, CACHE)

OUT = ROOT / "results" / "model_comparison"; PRED = CACHE / "preds" / "post_departure"; PRED.mkdir(parents=True, exist_ok=True)
df = load_features()
dep = connect().execute("SELECT flight_key, departure_delay_min AS dep_delay FROM fact_flight").fetchdf()
df = df.merge(dep, on="flight_key", how="left", validate="one_to_one"); df["dep_delay"] = df["dep_delay"].astype("float32")
tr, va, te = split_masks(df); y = df[TARGET].to_numpy(np.int8); ytr, yva, yte = y[tr], y[va], y[te]
FEATS = SCENARIOS["t60"] + ["dep_delay"]
print(f"rows {tr.sum():,}/{va.sum():,}/{te.sum():,}; dep_delay missing {df.dep_delay.isna().mean():.4%}; corr(dep_delay, target) = {np.corrcoef(df.dep_delay.fillna(0), y)[0,1]:.3f}")
CW = class_weights(ytr); W = np.where(y == 1, CW[1], CW[0]).astype("float32")
res = {}


def f2(p, r): return 5 * p * r / (4 * p + r) if p + r else 0.0


# ---- LightGBM (tuned params from Step 4) ----------------------------------------------------------------------
lp = json.load(open(OUT / "lgbm_best_params.json"))["best_params"]; cat = [c for c in FEATS if c in CAT_COLS]
t0 = time.time()
dtr = lgb.Dataset(df.loc[tr, FEATS], ytr, weight=W[tr], categorical_feature=cat, free_raw_data=False)
dva = lgb.Dataset(df.loc[va, FEATS], yva, categorical_feature=cat, reference=dtr)
bst = lgb.train({**LGB_BASE, **lp, "seed": SEED, "num_threads": 10}, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
res["LightGBM"] = (bst.predict(df.loc[va, FEATS], num_iteration=bst.best_iteration), bst.predict(df.loc[te, FEATS], num_iteration=bst.best_iteration), time.time() - t0)
print("LightGBM done", round(time.time() - t0), "s", flush=True)

# ---- XGBoost (tuned params from Step 4, ordinal-coded categoricals) -------------------------------------------
xp = json.load(open(OUT / "xgb_best_params.json"))["best_params"]; X = ordinal_codes(df, FEATS); t0 = time.time()
m = make_xgb({**XGB_DEFAULT, **xp}, SEED); m.fit(X[tr], ytr, sample_weight=W[tr], eval_set=[(X[va], yva)], sample_weight_eval_set=[W[va]], verbose=False)
res["XGBoost (tuned)"] = (m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1], time.time() - t0)
print("XGBoost done", round(time.time() - t0), "s", flush=True)

# ---- evaluation: F1-tuned threshold and a recall-matched (paper-style) operating point -------------------------
rows = []
for name, (sv, st, tt) in res.items():
    np.savez(PRED / f"{name.split()[0].lower()}__{SEED}.npz", val=sv.astype("float32"), test=st.astype("float32"))
    thr, _ = best_f1_threshold(yva, sv); a = metrics_at(yte, st, thr)
    thr83 = float(np.sort(sv[yva == 1])[int(0.17 * (yva == 1).sum())]); b = metrics_at(yte, st, thr83)   # ~83% recall on validation
    rows.append(dict(model=name, pr_auc=a["pr_auc"], roc_auc=a["roc_auc"], accuracy=a["accuracy"], precision=a["precision"], recall=a["recall"],
                     f1=a["f1"], f2=f2(a["precision"], a["recall"]), acc_at_recall83=b["accuracy"], prec_at_recall83=b["precision"],
                     rec_at_recall83=b["recall"], f2_at_recall83=f2(b["precision"], b["recall"]), train_s=round(tt, 1)))
out = pd.DataFrame(rows); out.to_csv(OUT / "post_departure_scenario.csv", index=False)
print(out.round(4).to_string(index=False))
