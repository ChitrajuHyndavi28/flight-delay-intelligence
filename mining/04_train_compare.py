"""Step 4 - predictive mining (O5): LightGBM (main) vs baselines on the T-60 operational feature set.

Stages (resumable; every model's val/test scores are cached in mining_cache/preds/):
  python 04_train_compare.py                 # everything
  python 04_train_compare.py --stages tune main ablation dayahead scalability evaluate
  python 04_train_compare.py --smoke         # tiny sample, separate cache: code/timing check only
Design: time-based split (train Jan-Aug / val Sep / test Nov-Dec, October excluded), balanced class weights computed
on train and used by every model that accepts weights, threshold = max-F1 on validation applied to test,
LightGBM tuned on validation only, 3 seeds for stochastic models, bootstrap CIs on test PR-AUC (evaluate stage).
"""
import argparse, itertools, json, os, sys, time, warnings
import numpy as np, pandas as pd
from ml_common import *
from ml_common import LGB_BASE as LGB_BASE_SHARED
from common import outdir, set_plot_style, SEED

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--stages", nargs="*", default=["tune", "main", "ablation", "dayahead", "scalability", "evaluate"])
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--models", nargs="*", default=None)
args = ap.parse_args()
NJOBS = 10
PRED = CACHE / ("preds_smoke" if args.smoke else "preds"); PRED.mkdir(exist_ok=True)
OUT = outdir("model_comparison_smoke" if args.smoke else "model_comparison")
SEEDS = [42, 43, 44]
RF_SAMPLE, KNN_SAMPLE, TUNE_SAMPLE = (20_000, 5_000, 20_000) if args.smoke else (1_000_000, 200_000, 1_000_000)

# --------------------------------------------------------------------------------------------------------- data
df = load_features()
if args.smoke:                                       # keep structure, shrink rows
    df = pd.concat([d.sample(min(len(d), n), random_state=0) for d, n in
                    [(df[df.split == "train"], 60_000), (df[df.split == "val"], 20_000), (df[df.split == "test"], 20_000)]])
tr, va, te = split_masks(df)
y = df[TARGET].to_numpy(np.int8); ytr, yva, yte = y[tr], y[va], y[te]
CW = class_weights(ytr); W_ALL = np.where(y == 1, CW[1], CW[0]).astype("float32")
print(f"rows train/val/test = {tr.sum():,}/{va.sum():,}/{te.sum():,}; delayed rate {ytr.mean():.3f}/{yva.mean():.3f}/{yte.mean():.3f}; class weights {CW}")
assert_no_leakage(ALL_FEATURES)
CARD = {c: len(df[c].cat.categories) for c in CAT_COLS}

def feats_of(scn): return [c for c in SCENARIOS[scn]]
def X_native(feats):   return df[feats]                                                # category dtype (LightGBM / XGB-native)
def X_codes(feats): return ordinal_codes(df, feats)

def stratified_idx_(mask, n, seed): return stratified_idx(y, mask, n, seed)      # positions in df; labels from the full vector y
_si = stratified_idx_

def save_run(scn, name, seed, val, test, train_time, pred_time, n_train, note="", extra=None):
    d = PRED / scn; d.mkdir(exist_ok=True)
    np.savez(d / f"{name}__{seed}.npz", val=np.asarray(val, "float32"), test=np.asarray(test, "float32"))
    json.dump(dict(model=name, scenario=scn, seed=seed, train_time_s=round(train_time, 2), predict_time_s=round(pred_time, 2),
                   n_train=int(n_train), note=note, **(extra or {})), open(d / f"{name}__{seed}.json", "w"))
def done(scn, name, seed): return (PRED / scn / f"{name}__{seed}.json").exists()

# ---------------------------------------------------------------------------------------------------- LightGBM
import lightgbm as lgb
LGB_BASE = {**LGB_BASE_SHARED, "num_threads": NJOBS}
BEST_FILE = OUT / "lgbm_best_params.json"

def lgb_fit(feats, params, idx, seed, rounds=3000, early=True):
    cat = [c for c in feats if c in CAT_COLS]
    Xtr = df[feats].iloc[idx]; dtr = lgb.Dataset(Xtr, ytr_full[idx], weight=W_ALL[idx], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(df[feats][va], yva, categorical_feature=cat, reference=dtr)
    t = time.time()
    bst = lgb.train({**LGB_BASE, **params, "seed": seed}, dtr, rounds, valid_sets=[dva],
                    callbacks=([lgb.early_stopping(50, verbose=False)] if early else []))
    return bst, time.time() - t

ytr_full = y            # full-length label vector (indices are positions in df)
TR_IDX = np.nonzero(tr)[0]

def stage_tune():
    if BEST_FILE.exists(): print("tuning cached:", json.load(open(BEST_FILE))["best_params"]); return
    feats = feats_of("t60"); idx = _si(tr, TUNE_SAMPLE, SEED)
    grid = list(itertools.product([63, 255], [0.1, 0.05], [100, 1000], [0.6, 0.9]))
    if args.smoke: grid = grid[:4]
    rows = []
    for nl, lr, mcs, ff in grid:
        p = dict(num_leaves=nl, learning_rate=lr, min_child_samples=mcs, feature_fraction=ff)
        bst, t = lgb_fit(feats, p, idx, SEED)
        s = bst.predict(df[feats][va], num_iteration=bst.best_iteration)
        rows.append({**p, "best_iter": bst.best_iteration, "val_pr_auc": average_precision_score(yva, s), "val_roc_auc": roc_auc_score(yva, s), "fit_s": round(t, 1)})
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False); g.to_csv(OUT / "lgbm_tuning_grid_validation.csv", index=False)
    best = g.iloc[0]; bp = dict(num_leaves=int(best.num_leaves), learning_rate=float(best.learning_rate), min_child_samples=int(best.min_child_samples), feature_fraction=float(best.feature_fraction))
    json.dump(dict(best_params=bp, tuned_on=f"{len(idx):,}-row stratified train sample, selected on validation PR-AUC only"), open(BEST_FILE, "w"))
    print("best:", bp)

def lgbm_run(scn, seed, name="lightgbm", idx=None, note=""):
    if done(scn, name, seed): return
    idx = TR_IDX if idx is None else idx
    feats = feats_of(scn); bp = json.load(open(BEST_FILE))["best_params"]
    bst, t = lgb_fit(feats, bp, idx, seed)
    t0 = time.time(); st = bst.predict(df[feats][te], num_iteration=bst.best_iteration); pt = time.time() - t0
    sv = bst.predict(df[feats][va], num_iteration=bst.best_iteration)
    save_run(scn, name, seed, sv, st, t, pt, len(idx), note=note, extra=dict(best_iter=bst.best_iteration))
    if scn == "t60" and seed == SEED and name == "lightgbm": bst.save_model(str(CACHE / ("lgbm_smoke.txt" if args.smoke else "lgbm_t60_seed42.txt")), num_iteration=bst.best_iteration)
    print(f"[{scn}] {name} seed {seed}: val PR-AUC {average_precision_score(yva, sv):.4f} test PR-AUC {average_precision_score(yte, st):.4f} train {t:.0f}s iters {bst.best_iteration}", flush=True)

# ------------------------------------------------------------------------------------------------------ baselines
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
DT_FILE = OUT / "dt_best_depth.json"

def predict_proba1(model, X, chunk=None):
    if chunk is None: return model.predict_proba(X)[:, 1]
    return np.concatenate([model.predict_proba(X[i:i + chunk])[:, 1] for i in range(0, X.shape[0], chunk)])

def fit_sk(name, scn, seed):
    """Fit one sklearn baseline; returns (val_scores, test_scores, train_time, predict_time, n_train, note)."""
    feats = feats_of(scn); Xdf = X_codes(feats)
    if name == "dummy_majority":
        t0 = time.time(); st = np.zeros(te.sum()); return np.zeros(va.sum()), st, 0.0, time.time() - t0, len(TR_IDX), "predicts 'not delayed' always"
    dt_depth = json.load(open(DT_FILE))["best_depth"] if name == "decision_tree" else 12
    spec = sk_model_spec(name, seed, NJOBS, dt_depth=dt_depth, card=CARD, feats=feats); pre = sk_pre(feats, spec["kind"]); m = spec["model"]
    idx = {"knn": lambda: _si(tr, KNN_SAMPLE, seed), "random_forest": lambda: _si(tr, RF_SAMPLE, seed)}.get(name, lambda: TR_IDX)()
    note = {"knn": f"stratified {len(idx):,}-row train sample; k=50, uniform weights (no class-weight support); numeric features only",
            "random_forest": f"stratified {len(idx):,}-row train sample (full data infeasible); 150 trees, min_samples_leaf=50",
            "decision_tree": f"depth {dt_depth} tuned on validation", "logistic_regression": "one-hot (min freq 200) + scaled numerics, median imputation + missing indicators",
            "naive_bayes_gaussian": "numeric features only; balanced priors", "naive_bayes_categorical": "10-quantile-binned numerics + categorical codes; balanced priors"}[name]
    cast = (lambda a: a.astype("int32")) if spec["kind"] == "catnb" else (lambda a: a)
    t = time.time(); Xt = cast(pre.fit_transform(Xdf.iloc[idx]))
    m.fit(Xt, y[idx], sample_weight=W_ALL[idx]) if spec["weighted"] else m.fit(Xt, y[idx]); tt = time.time() - t
    chunk = 20_000 if name == "knn" else None
    t0 = time.time(); st = predict_proba1(m, cast(pre.transform(Xdf[te])), chunk); pt = time.time() - t0
    sv = predict_proba1(m, cast(pre.transform(Xdf[va])), chunk)
    return sv, st, tt, pt, len(idx), note

def tune_dt():
    if DT_FILE.exists(): return
    feats = feats_of("t60"); Xdf = X_codes(feats); idx = _si(tr, TUNE_SAMPLE, SEED); pre = sk_pre(feats, "tree")
    Xt = pre.fit_transform(Xdf.iloc[idx]); Xv = pre.transform(Xdf[va]); rows = []
    for d in ([3, 5] if args.smoke else [3, 4, 5, 6, 8, 10, 12, 14, 16, 20]):
        m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, random_state=SEED).fit(Xt, y[idx], sample_weight=W_ALL[idx])
        s = m.predict_proba(Xv)[:, 1]; rows.append(dict(max_depth=d, val_pr_auc=average_precision_score(yva, s), val_roc_auc=roc_auc_score(yva, s))); print(rows[-1], flush=True)
    g = pd.DataFrame(rows); g.to_csv(OUT / "dt_depth_tuning_validation.csv", index=False)
    json.dump(dict(best_depth=int(g.sort_values("val_pr_auc", ascending=False).iloc[0].max_depth), tuned_on=f"{len(idx):,}-row stratified train sample, validation PR-AUC"), open(DT_FILE, "w"))

XGB_BEST_FILE = OUT / "xgb_best_params.json"
def xgb_run(scn, seed, native=False, name=None, tuned=False, idx=None, note=""):
    name = name or (("xgboost_native_tuned" if tuned else "xgboost_native_cat") if native else ("xgboost_tuned" if tuned else "xgboost"))
    if done(scn, name, seed): return
    params = json.load(open(XGB_BEST_FILE))["best_params"] if tuned else XGB_DEFAULT
    idx = TR_IDX if idx is None else idx
    feats = feats_of(scn); X = X_native(feats) if native else X_codes(feats)
    m = make_xgb(params, seed, native, NJOBS)
    t = time.time(); m.fit(X.iloc[idx], y[idx], sample_weight=W_ALL[idx], eval_set=[(X[va], yva)], verbose=False); tt = time.time() - t
    t0 = time.time(); st = m.predict_proba(X[te])[:, 1]; pt = time.time() - t0; sv = m.predict_proba(X[va])[:, 1]
    save_run(scn, name, seed, sv, st, tt, pt, len(idx), note=(("native categoricals" if native else "ordinal-coded categoricals, native NaN") + ("; tuned" if tuned else "; default params") + ("; " + note if note else "")), extra=dict(best_iter=int(m.best_iteration)))
    print(f"[{scn}] {name} seed {seed}: val PR-AUC {average_precision_score(yva, sv):.4f} test {average_precision_score(yte, st):.4f} train {tt:.0f}s", flush=True)

def sk_run(scn, name, seed):
    if done(scn, name, seed): return
    t = time.time(); sv, st, tt, pt, n, note = fit_sk(name, scn, seed)
    save_run(scn, name, seed, sv, st, tt, pt, n, note=note)
    print(f"[{scn}] {name} seed {seed}: val PR-AUC {average_precision_score(yva, sv):.4f} test {average_precision_score(yte, st):.4f} train {tt:.0f}s predict {pt:.0f}s (wall {time.time()-t:.0f}s)", flush=True)

SK_SEEDED = ["random_forest", "knn", "decision_tree"]; SK_DET = ["dummy_majority", "logistic_regression", "naive_bayes_gaussian", "naive_bayes_categorical"]
def wanted(m): return args.models is None or m in args.models

def stage_main(scn="t60"):
    tune_dt()
    for s in SEEDS[:1 if args.smoke else 3]:
        if wanted("lightgbm"): lgbm_run(scn, s)
    for m in SK_DET:
        if wanted(m): sk_run(scn, m, SEED)
    for s in SEEDS[:1 if args.smoke else 3]:
        if wanted("xgboost"): xgb_run(scn, s)
        for m in SK_SEEDED:
            if wanted(m): sk_run(scn, m, s)

def stage_ablation():
    if wanted("lightgbm"):
        for s in SEEDS[:1 if args.smoke else 3]: lgbm_run("dayahead", s)
        for scn in ["dayahead_plus_sched", "t60_no_weather", "t60_no_month", "t60_no_hist"]: lgbm_run(scn, SEED)
    if wanted("xgboost_native_cat"): xgb_run("t60", SEED, native=True)

def stage_dayahead():
    if wanted("xgboost"): xgb_run("dayahead", SEED)
    for m in SK_DET + SK_SEEDED:
        if wanted(m): sk_run("dayahead", m, SEED)


def stage_tune_xgb():
    """Give XGBoost the same tuning budget as LightGBM: 16 configurations, 1M stratified train sample, selected on validation PR-AUC only."""
    if XGB_BEST_FILE.exists(): print("xgb tuning cached:", json.load(open(XGB_BEST_FILE))["best_params"]); return
    feats = feats_of("t60"); X = X_codes(feats); idx = _si(tr, TUNE_SAMPLE, SEED); rows = []
    grid = list(itertools.product([6, 8], [0.1, 0.05], [5, 50], [0.6, 0.9])); grid = grid[:4] if args.smoke else grid
    for md, lr, mcw, cs in grid:
        p = dict(max_depth=md, learning_rate=lr, min_child_weight=mcw, colsample_bytree=cs)
        m = make_xgb(p, SEED, False, NJOBS)
        t = time.time(); m.fit(X.iloc[idx], y[idx], sample_weight=W_ALL[idx], eval_set=[(X[va], yva)], verbose=False); tt = time.time() - t
        sv = m.predict_proba(X[va])[:, 1]; rows.append({**p, "best_iter": int(m.best_iteration), "val_pr_auc": average_precision_score(yva, sv), "val_roc_auc": roc_auc_score(yva, sv), "fit_s": round(tt, 1)}); print(rows[-1], flush=True)
    g = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False); g.to_csv(OUT / "xgb_tuning_grid_validation.csv", index=False); b = g.iloc[0]
    bp = dict(max_depth=int(b.max_depth), learning_rate=float(b.learning_rate), min_child_weight=int(b.min_child_weight), colsample_bytree=float(b.colsample_bytree))
    json.dump(dict(best_params=bp, tuned_on=f"{len(idx):,}-row stratified train sample, validation PR-AUC only"), open(XGB_BEST_FILE, "w")); print("best:", bp)

def stage_xgb_tuned():
    stage_tune_xgb()
    for s in SEEDS[:1 if args.smoke else 3]: xgb_run("t60", s, tuned=True)
    xgb_run("t60", SEED, native=True, tuned=True)
    xgb_run("dayahead", SEED, tuned=True)
    xgb_run("dayahead", SEED, native=True, tuned=True)

def stage_matched():
    """Same algorithm, same rows: LightGBM / tuned XGBoost refit on the EXACT samples used by Random Forest (1M) and KNN (200k)."""
    for s in SEEDS[:1 if args.smoke else 3]:
        for tag, n in [("rf1M", RF_SAMPLE), ("knn200k", KNN_SAMPLE)]:
            idx = _si(tr, n, s)       # identical function + seed => identical rows to fit_sk('random_forest' / 'knn')
            if wanted("lightgbm"): lgbm_run("t60", s, name=f"lightgbm_matched_{tag}", idx=idx, note=f"same {len(idx):,} stratified rows as the {'Random Forest' if tag=='rf1M' else 'KNN'} baseline; hyperparameters not re-tuned")
            if wanted("xgboost_tuned"): xgb_run("t60", s, tuned=True, name=f"xgboost_tuned_matched_{tag}", idx=idx, note=f"same {len(idx):,} rows as the {'Random Forest' if tag=='rf1M' else 'KNN'} baseline")

def stage_ablation2():
    if wanted("lightgbm"): lgbm_run("dayahead_no_weather", SEED)

# ------------------------------------------------------------------------------------------------ scalability
def stage_scalability():
    fn = OUT / "scalability.csv"; rows = [] if not fn.exists() else pd.read_csv(fn).to_dict("records"); have = {(r["model"], r["n_train"]) for r in rows}
    feats = feats_of("t60"); bp = json.load(open(BEST_FILE))["best_params"]; Xc = X_codes(feats)
    sizes = [5_000, 10_000, 20_000, len(TR_IDX)] if args.smoke else [100_000, 500_000, 1_000_000, len(TR_IDX)]
    for n in sizes:
        idx = _si(tr, n, SEED)
        def rec(model, t, sv, note=""):
            r = dict(model=model, n_train=len(idx), train_time_s=round(t, 1), val_pr_auc=round(average_precision_score(yva, sv), 4), note=note); rows.append(r); print(r, flush=True)
            pd.DataFrame(rows).to_csv(fn, index=False)
        if wanted("lightgbm") and ("lightgbm", len(idx)) not in have:
            cat = [c for c in feats if c in CAT_COLS]; dtr = lgb.Dataset(df[feats].iloc[idx], y[idx], weight=W_ALL[idx], categorical_feature=cat)
            t = time.time(); b = lgb.train({**LGB_BASE, **bp, "seed": SEED}, dtr, 300); tt = time.time() - t; rec("lightgbm", tt, b.predict(df[feats][va]), "300 rounds, tuned params, no early stopping")
        if wanted("xgboost") and ("xgboost", len(idx)) not in have:
            m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, tree_method="hist", n_jobs=NJOBS, random_state=SEED)
            t = time.time(); m.fit(Xc.iloc[idx], y[idx], sample_weight=W_ALL[idx]); tt = time.time() - t; rec("xgboost", tt, m.predict_proba(Xc[va])[:, 1], "300 rounds, depth 8")
        if wanted("logistic_regression") and ("logistic_regression", len(idx)) not in have:
            pre = sk_pre(feats, "linear"); t = time.time(); Xt = pre.fit_transform(Xc.iloc[idx]); m = LogisticRegression(max_iter=200).fit(Xt, y[idx], sample_weight=W_ALL[idx]); tt = time.time() - t
            rec("logistic_regression", tt, m.predict_proba(pre.transform(Xc[va]))[:, 1], "includes one-hot preprocessing")
        if wanted("random_forest") and ("random_forest", len(idx)) not in have and (len(idx) <= 1_000_000 or args.smoke):
            pre = sk_pre(feats, "tree"); t = time.time(); Xt = pre.fit_transform(Xc.iloc[idx]); m = RandomForestClassifier(150, min_samples_leaf=50, max_features="sqrt", n_jobs=NJOBS, random_state=SEED).fit(Xt, y[idx], sample_weight=W_ALL[idx]); tt = time.time() - t
            rec("random_forest", tt, m.predict_proba(pre.transform(Xc[va]))[:, 1], "150 trees")


def stage_cluster_boot():
    """Day-clustered bootstrap: resample whole test DAYS (flights on one day share weather/congestion/aircraft), so the
    flight-level Poisson CIs above are optimistic. Reports paired PR-AUC differences vs LightGBM with day-clustered 95% CIs."""
    days = df.date_key.to_numpy()[te]; ud, inv = np.unique(days, return_inverse=True); nd = len(ud); rng = np.random.default_rng(SEED)
    B = PoissonBootstrapAP(yte, 1000); rows = []
    for scn, others in [("t60", ["xgboost_tuned", "xgboost_native_tuned", "xgboost", "random_forest", "decision_tree", "logistic_regression"]), ("dayahead", ["xgboost_tuned", "xgboost_native_tuned", "random_forest"])]:
        names = ["lightgbm"] + [o for o in others if (PRED / scn / f"{o}__{SEED}.npz").exists()]
        preps = {k: B.prepare(np.load(PRED / scn / f"{k}__{SEED}.npz")["test"]) for k in names}; res = {k: np.empty(1000) for k in names}
        for b in range(1000):
            w = np.bincount(rng.integers(0, nd, nd), minlength=nd)[inv].astype(np.float32)
            for k in names: res[k][b] = B.ap(preps[k], w)
        for k in names[1:]:
            d = res["lightgbm"] - res[k]; lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(scenario=scn, comparison=f"LightGBM minus {LABEL[k]}", mean_diff_pr_auc=d.mean(), day_cluster_ci_lo=lo, day_cluster_ci_hi=hi, n_days=nd, verdict=verdict(lo, hi, d.mean())))
    R = pd.DataFrame(rows); R.to_csv(OUT / "cluster_bootstrap_by_day.csv", index=False); print(R.round(4).to_string(index=False))

# --------------------------------------------------------------------------------------------------- evaluate
def load_runs(scn):
    runs = []
    for js in sorted((PRED / scn).glob("*.json")):
        meta = json.load(open(js)); z = np.load(js.with_suffix(".npz")); runs.append((meta, z["val"], z["test"]))
    return runs

LABEL = {"lightgbm": "LightGBM (main)", "xgboost": "XGBoost (default params, ordinal codes)", "xgboost_tuned": "XGBoost (tuned, ordinal codes)",
         "xgboost_native_cat": "XGBoost (native categoricals, default params)", "xgboost_native_tuned": "XGBoost (native categoricals, tuned)",
         "random_forest": "Random Forest", "decision_tree": "Decision Tree (CART)", "logistic_regression": "Logistic Regression",
         "naive_bayes_gaussian": "Naive Bayes (Gaussian)", "naive_bayes_categorical": "Naive Bayes (Categorical)", "knn": "KNN", "dummy_majority": "Majority dummy",
         "lightgbm_matched_rf1M": "LightGBM on RF's 1M sample", "lightgbm_matched_knn200k": "LightGBM on KNN's 200k sample",
         "xgboost_tuned_matched_rf1M": "XGBoost (tuned) on RF's 1M sample", "xgboost_tuned_matched_knn200k": "XGBoost (tuned) on KNN's 200k sample"}
is_matched = lambda k: "matched" in k

def oracle_f1(y_, s_):
    p_, r_, _ = precision_recall_curve(y_, s_); return float(np.max(2 * p_ * r_ / np.clip(p_ + r_, 1e-12, None)))

def comparison_table(scn):
    rows, per = [], {}
    for meta, sv, st in load_runs(scn):
        const = sv.std() == 0
        thr, fval = (1.1, 0.0) if const else best_f1_threshold(yva, sv)
        m = dict(roc_auc=0.5, pr_auc=yte.mean(), f1=0.0, precision=0.0, recall=0.0, accuracy=1 - yte.mean(), predicted_positive_rate=0.0) if const else metrics_at(yte, st, thr)
        m.update(model=meta["model"], seed=meta["seed"], threshold=thr, val_f1=fval, oracle_f1=0.0 if const else oracle_f1(yte, st),
                 val_pr_auc=yva.mean() if const else average_precision_score(yva, sv), train_time_s=meta["train_time_s"], predict_time_s=meta["predict_time_s"], n_train=meta["n_train"], note=meta["note"])
        rows.append(m); per[(meta["model"], meta["seed"])] = (sv, st)
    R = pd.DataFrame(rows); R.to_csv(OUT / f"all_runs_{scn}.csv", index=False); agg = []
    for mdl, g in R.groupby("model"):
        r = dict(model=LABEL.get(mdl, mdl), key=mdl, n_seeds=len(g), n_train=int(g.n_train.iloc[0]), note=g.note.iloc[0])
        for c in ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "train_time_s", "predict_time_s", "threshold", "val_pr_auc", "val_f1", "oracle_f1", "predicted_positive_rate"]:
            r[c] = g[c].mean(); r[c + "_std"] = g[c].std(ddof=0) if len(g) > 1 else np.nan
        r["lift"] = r["pr_auc"] / yte.mean(); agg.append(r)
    return pd.DataFrame(agg).sort_values("pr_auc", ascending=False), per

def fmt(a, c, p=4): return a[c].map(lambda v: f"{v:.{p}f}") + a[c + "_std"].map(lambda v: "" if pd.isna(v) else f" ± {v:.{p}f}")
def ci_str(lo, hi): return f"[{lo:.4f}, {hi:.4f}]"
def verdict(lo, hi, mean, margin=0.01):
    if lo > 0 or hi < 0:
        return "statistically real" + (" but practically negligible (< 0.01)" if abs(mean) < margin else " and practically meaningful")
    return "not statistically distinguishable" + (" (practically equivalent, CI inside +/-0.01)" if max(abs(lo), abs(hi)) < margin else " (CI too wide to call equivalent)")

def stage_evaluate():
    import matplotlib.pyplot as plt_; set_plot_style()
    nboot = 200 if args.smoke else 1000; base = float(yte.mean())
    for scn in ["t60", "dayahead"]:
        if not (PRED / scn).exists(): continue
        A, per = comparison_table(scn); keys = [k for k in A.key if (k, SEED) in per and k != "dummy_majority"]
        bs = PoissonBootstrapAP(yte, nboot).run({k: per[(k, SEED)][1] for k in keys})["ap"]
        ci = pd.DataFrame({k: dict(lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)) for k, v in bs.items()}).T
        A = A.merge(ci, left_on="key", right_index=True, how="left"); A.loc[A.key == "dummy_majority", ["lo", "hi"]] = base
        M = A[~A.key.map(is_matched)]; top = M[M.key != "dummy_majority"].iloc[0].key
        T = pd.DataFrame({"Model": M.model, "Test base rate": f"{base:.4f}", "PR-AUC (primary)": fmt(M, "pr_auc"), "Lift over base rate": M.lift.map("{:.2f}x".format),
                          "PR-AUC boot 95% CI": M.apply(lambda r: ci_str(r.lo, r.hi), axis=1), "ROC-AUC": fmt(M, "roc_auc"), "F1 @ val-tuned thr": fmt(M, "f1"),
                          "Precision": fmt(M, "precision"), "Recall": fmt(M, "recall"), "Val F1 (tuning)": fmt(M, "val_f1"), "Oracle test F1 (diagnostic only)": fmt(M, "oracle_f1"),
                          "Accuracy (misleading)": fmt(M, "accuracy"), "Train s": fmt(M, "train_time_s", 1), "Predict s": fmt(M, "predict_time_s", 1), "Train rows": M.n_train.map("{:,}".format), "Notes": M.note})
        T.to_csv(OUT / f"model_comparison_{scn}.csv", index=False); A.to_csv(OUT / f"model_comparison_{scn}_numeric.csv", index=False)
        print(f"\n===== {scn} (test base rate {base:.4f}, val base rate {yva.mean():.4f}) =====\n", T.drop(columns=["Notes", "Oracle test F1 (diagnostic only)", "Val F1 (tuning)"]).to_string(index=False))
        # paired differences vs top
        D = pd.DataFrame([dict(top_model=LABEL.get(top, top), vs=LABEL.get(k, k), mean_diff_pr_auc=(bs[top] - bs[k]).mean(), ci_lo=np.percentile(bs[top] - bs[k], 2.5), ci_hi=np.percentile(bs[top] - bs[k], 97.5),
                          verdict=verdict(np.percentile(bs[top] - bs[k], 2.5), np.percentile(bs[top] - bs[k], 97.5), (bs[top] - bs[k]).mean())) for k in keys if k != top and not is_matched(k)]).sort_values("mean_diff_pr_auc")
        D.to_csv(OUT / f"paired_bootstrap_vs_top_{scn}.csv", index=False); print(D.round(4).to_string(index=False))
        # LightGBM vs XGBoost
        if "lightgbm" in bs:
            X_ = []
            for k in ["xgboost_tuned", "xgboost", "xgboost_native_tuned", "xgboost_native_cat"]:
                if k in bs:
                    d = bs["lightgbm"] - bs[k]; r_a, r_b = A[A.key == "lightgbm"].iloc[0], A[A.key == k].iloc[0]
                    X_.append(dict(comparison=f"LightGBM minus {LABEL[k]}", mean_diff_pr_auc=d.mean(), ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), verdict=verdict(np.percentile(d, 2.5), np.percentile(d, 97.5), d.mean()),
                                   lightgbm_train_s=r_a.train_time_s, other_train_s=r_b.train_time_s, lightgbm_predict_s=r_a.predict_time_s, other_predict_s=r_b.predict_time_s, other_seed_std_pr_auc=r_b.pr_auc_std, lightgbm_seed_std_pr_auc=r_a.pr_auc_std))
            pd.DataFrame(X_).to_csv(OUT / f"lightgbm_vs_xgboost_{scn}.csv", index=False); print(pd.DataFrame(X_).round(4).to_string(index=False))
        # matched-data comparison
        pairs = [("1M stratified sample", "lightgbm_matched_rf1M", "xgboost_tuned_matched_rf1M", "random_forest"), ("200k stratified sample", "lightgbm_matched_knn200k", "xgboost_tuned_matched_knn200k", "knn")]
        MR = []
        for lab, kl, kx, kb in pairs:
            for k in [kl, kx, kb]:
                if k in A.key.values:
                    r = A[A.key == k].iloc[0]; MR.append(dict(training_data=lab, model=LABEL.get(k, k), pr_auc=f"{r.pr_auc:.4f}" + (f" ± {r.pr_auc_std:.4f}" if pd.notna(r.pr_auc_std) else ""), lift=f"{r.lift:.2f}x", pr_auc_boot_ci=ci_str(r.lo, r.hi),
                                                              roc_auc=f"{r.roc_auc:.4f}", train_s=round(r.train_time_s, 1), n_train=int(r.n_train)))
            if kl in bs and kb in bs:
                d = bs[kl] - bs[kb]; MR.append(dict(training_data=lab, model=f"  >> LightGBM minus {LABEL[kb]} (paired)", pr_auc=f"{d.mean():+.4f}", pr_auc_boot_ci=ci_str(np.percentile(d, 2.5), np.percentile(d, 97.5)), lift=verdict(np.percentile(d, 2.5), np.percentile(d, 97.5), d.mean())))
        if MR and scn == "t60":
            full = A[A.key == "lightgbm"].iloc[0]; MR.append(dict(training_data="full training set (3.84M rows)", model=LABEL["lightgbm"], pr_auc=f"{full.pr_auc:.4f} ± {full.pr_auc_std:.4f}", lift=f"{full.lift:.2f}x", pr_auc_boot_ci=ci_str(full.lo, full.hi), roc_auc=f"{full.roc_auc:.4f}", train_s=round(full.train_time_s, 1), n_train=int(full.n_train)))
            pd.DataFrame(MR).to_csv(OUT / "matched_training_data_comparison.csv", index=False); print("\nMatched-data comparison:\n", pd.DataFrame(MR).to_string(index=False))
        # chart
        a = M.iloc[::-1]; fig, ax = plt_.subplots(figsize=(11, 6.5))
        ax.barh(a.model, a.pr_auc, xerr=[a.pr_auc - a.lo, a.hi - a.pr_auc], color=["#2a7ab0" if k == top else "#9aa5b1" for k in a.key], capsize=3)
        ax.axvline(base, color="#b03a2e", ls="--", label=f"Test base rate ({base:.3f}); lift = PR-AUC / base rate"); ax.set_xlabel("Test PR-AUC (bootstrap 95% CI, 1,000 resamples)"); ax.set_ylabel("Model")
        ax.set_title(f"Model comparison, test set ({'T-60 operational' if scn == 't60' else 'day-ahead'} features)"); ax.legend(frameon=False); fig.savefig(OUT / f"model_comparison_{scn}.png"); plt_.close(fig)
        fig, ax = plt_.subplots(figsize=(8, 6))
        for k in [k for k in keys if not is_matched(k)]:
            p_, r_, _ = precision_recall_curve(yte, per[(k, SEED)][1]); ax.plot(r_, p_, label=LABEL.get(k, k), lw=2.5 if k == top else 1.1)
        ax.axhline(base, color="#b03a2e", ls=":"); ax.set_xlabel("Recall (delayed class)"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves, test set"); ax.legend(frameon=False, fontsize=7); fig.savefig(OUT / f"pr_curves_{scn}.png"); plt_.close(fig)
    # ------------------------------------------------ ablation (LightGBM, seed-42 paired bootstrap, PR-AUC and ROC-AUC)
    order = [("dayahead_no_weather", "Day-ahead, weather OFF"), ("dayahead", "Day-ahead, weather ON"), ("dayahead_plus_sched", "Day-ahead + schedule-known turn structure (weather ON)"),
             ("t60_no_weather", "T-60, weather OFF"), ("t60", "T-60, weather ON  [main]"), ("t60_no_month", "T-60 without month / week_of_year"), ("t60_no_hist", "T-60 without historical delay rates")]
    sc, ms = {}, {}
    for scn, lab in order:
        runs = [(m, sv, st) for m, sv, st in load_runs(scn) if m["model"] == "lightgbm"] if (PRED / scn).exists() else []
        if not runs: continue
        sc[scn] = [r for r in runs if r[0]["seed"] == SEED][0][2]; M_ = []
        for m, sv, st in runs: thr, _ = best_f1_threshold(yva, sv); M_.append(metrics_at(yte, st, thr))
        ms[scn] = pd.DataFrame(M_)
    if sc:
        bb = PoissonBootstrapAP(yte, nboot).run(sc, ("ap", "auc")); rows = []
        for scn, lab in order:
            if scn not in sc: continue
            M_ = ms[scn]; pa, ro = bb["ap"][scn], bb["auc"][scn]
            rows.append(dict(setting=lab, n_features=len(SCENARIOS[scn]), seeds=len(M_), pr_auc=f"{M_.pr_auc.mean():.4f}" + (f" ± {M_.pr_auc.std(ddof=0):.4f}" if len(M_) > 1 else ""), lift=f"{M_.pr_auc.mean() / base:.2f}x",
                             pr_auc_ci=ci_str(*np.percentile(pa, [2.5, 97.5])), roc_auc=f"{M_.roc_auc.mean():.4f}" + (f" ± {M_.roc_auc.std(ddof=0):.4f}" if len(M_) > 1 else ""), roc_auc_ci=ci_str(*np.percentile(ro, [2.5, 97.5])),
                             f1=f"{M_.f1.mean():.4f}", recall=f"{M_.recall.mean():.4f}", precision=f"{M_.precision.mean():.4f}"))
        AB = pd.DataFrame(rows); AB.to_csv(OUT / "ablation_lightgbm.csv", index=False); print("\n===== ablation (LightGBM) =====\n", AB.to_string(index=False))
        eff = []
        for lab, x, y_ in [("Weather effect, day-ahead (ON minus OFF)", "dayahead", "dayahead_no_weather"), ("Weather effect, T-60 (ON minus OFF)", "t60", "t60_no_weather"),
                           ("Propagation effect: T-60 minus day-ahead (weather ON)", "t60", "dayahead"), ("  of which schedule-known turn structure", "dayahead_plus_sched", "dayahead"),
                           ("  of which live predecessor status / delays", "t60", "dayahead_plus_sched"), ("Month / week_of_year effect (with minus without)", "t60", "t60_no_month"), ("Historical-rate effect (with minus without)", "t60", "t60_no_hist")]:
            if x in sc and y_ in sc:
                dp, dr = bb["ap"][x] - bb["ap"][y_], bb["auc"][x] - bb["auc"][y_]
                eff.append(dict(effect=lab, delta_pr_auc=round(dp.mean(), 4), pr_auc_ci=ci_str(*np.percentile(dp, [2.5, 97.5])), delta_roc_auc=round(dr.mean(), 4), roc_auc_ci=ci_str(*np.percentile(dr, [2.5, 97.5]))))
        E = pd.DataFrame(eff); E.to_csv(OUT / "ablation_effects_paired.csv", index=False); print(E.to_string(index=False))
        fig, ax = plt_.subplots(figsize=(10, 5)); labs = [l for s_, l in order if s_ in sc][::-1]; v = [bb["ap"][s_] for s_, l in order if s_ in sc][::-1]; mu = np.array([x.mean() for x in v])
        lo = np.array([np.percentile(x, 2.5) for x in v]); hi = np.array([np.percentile(x, 97.5) for x in v])
        ax.barh(labs, mu, xerr=[mu - lo, hi - mu], color=["#2a7ab0" if "main" in l else "#9aa5b1" for l in labs], capsize=3); ax.axvline(base, color="#b03a2e", ls="--", label="Test base rate"); ax.set_xlabel("Test PR-AUC (bootstrap 95% CI)"); ax.set_ylabel("LightGBM feature set")
        ax.set_title("Ablation: what each feature family adds"); ax.legend(frameon=False); fig.savefig(OUT / "ablation_lightgbm.png"); plt_.close(fig)
    if (OUT / "scalability.csv").exists():
        S = pd.read_csv(OUT / "scalability.csv"); fig, axs = plt_.subplots(1, 2, figsize=(14, 5))
        for mdl, g in S.groupby("model"):
            g = g.sort_values("n_train"); axs[0].plot(g.n_train, g.train_time_s, "o-", label=LABEL.get(mdl, mdl)); axs[1].plot(g.n_train, g.val_pr_auc, "o-", label=LABEL.get(mdl, mdl))
        for ax in axs: ax.set_xscale("log"); ax.set_xlabel("Training rows"); ax.legend(frameon=False)
        axs[0].set_yscale("log"); axs[0].set_ylabel("Training time (s, log scale)"); axs[0].set_title("Scalability: training time"); axs[1].set_ylabel("Validation PR-AUC"); axs[1].set_title("Learning curve (validation)")
        fig.savefig(OUT / "scalability.png"); plt_.close(fig)

STAGES = {"cluster_boot": stage_cluster_boot, "tune_xgb": stage_tune_xgb, "xgb_tuned": stage_xgb_tuned, "matched": stage_matched, "ablation2": stage_ablation2, "tune": stage_tune, "main": stage_main, "ablation": stage_ablation, "dayahead": stage_dayahead, "scalability": stage_scalability, "evaluate": stage_evaluate}
for s in args.stages:
    t = time.time(); print(f"\n######## stage {s} ########", flush=True); STAGES[s](); print(f"stage {s} done in {time.time()-t:.0f}s", flush=True)
