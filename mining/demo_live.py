"""Live classroom demo (Review 2): the warehouse is real, the algorithm comparison is live, a live prediction.

    python mining/demo_live.py --stage 1|2|3|all

Stage 1  Warehouse is real      : row counts + 2 live OLAP queries (DuckDB, read-only)
Stage 2  Live algorithm compare : 9 models on a 200k-row train sample / 100k-row test sample, Step-4 hyperparameters, no tuning
Stage 3  Live prediction        : 5 real test flights scored by the saved full-data T-60 LightGBM model (models/)
The DuckDB connection is read-only throughout; the demo never modifies flight_dw.duckdb and never trains the full model.
Everything printed is also written to results/demo/demo_run_<timestamp>.txt.
"""
import argparse, datetime, io, json, sys, time, traceback, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from common import ROOT, SEED, connect, outdir
from ml_common import (ALL_FEATURES, CAT_COLS, LGB_BASE, TARGET, assert_no_leakage, best_f1_threshold, class_weights, load_features,
                       make_xgb, metrics_at, ordinal_codes, sk_model_spec, sk_pre, split_masks, stratified_idx)

OUT = outdir("demo"); TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); LOG = OUT / f"demo_run_{TS}.txt"
NJOBS = 10; TRAIN_N, TEST_N, VAL_N = 200_000, 100_000, 50_000
W = 96


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x): [f.write(x) for f in self.s]
    def flush(self): [f.flush() for f in self.s]


def banner(t): print("\n" + "=" * W + f"\n{t}\n" + "=" * W, flush=True)
def show(df): print(df.to_string(index=False), flush=True)

_FEATS = {}
def features():
    if "df" not in _FEATS:
        t = time.time(); _FEATS["df"] = load_features(); print(f"(loaded cached T-60 feature matrix: {len(_FEATS['df']):,} flights x {len(ALL_FEATURES)} features in {time.time() - t:.1f}s)", flush=True)
    return _FEATS["df"]


# ------------------------------------------------------------------------------------------------ stage 1
def stage1():
    banner("STAGE 1 - THE WAREHOUSE IS REAL   (DuckDB, read-only)")
    con = connect(read_only=True)
    rows = [(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in ["fact_flight", "dim_date", "dim_airline", "dim_airport", "dim_route", "dim_weather"]]
    print("Row counts:"); show(pd.DataFrame(rows, columns=["table", "rows"]).assign(rows=lambda d: d.rows.map("{:,}".format)))
    t = time.time()
    q1 = con.execute("""SELECT 'Q' || d.quarter AS quarter, count(*) AS flights,
         round(avg(CASE WHEN f.cancelled_flag=0 AND f.diverted_flag=0 THEN f.arrival_delay_min END), 1) AS avg_arrival_delay_min,
         round(100.0*sum(CASE WHEN f.cancelled_flag=0 AND f.diverted_flag=0 AND f.arrival_delay_min<15 THEN 1 ELSE 0 END)
               / sum(CASE WHEN f.cancelled_flag=0 AND f.diverted_flag=0 THEN 1 ELSE 0 END), 1) AS on_time_pct,
         sum(f.cancelled_flag) AS cancellations
       FROM fact_flight f JOIN dim_date d USING(date_key) GROUP BY 1 ORDER BY 1""").df()
    print(f"\nA. ROLL-UP: flights, delay, on-time % and cancellations by quarter        (query time {time.time() - t:.2f} s)")
    show(q1.assign(flights=q1.flights.map("{:,}".format), cancellations=q1.cancellations.map("{:,.0f}".format)))
    t = time.time()
    q2 = con.execute("""WITH top AS (SELECT airline_key FROM fact_flight GROUP BY 1 ORDER BY count(*) DESC LIMIT 8)
       SELECT l.airline_iata_code AS airline, d.month_number AS m, avg(f.arrival_delay_min) AS v
       FROM fact_flight f JOIN dim_date d USING(date_key) JOIN dim_airline l USING(airline_key)
       WHERE f.airline_key IN (SELECT airline_key FROM top) AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1,2""").df()
    pv = q2.pivot(index="airline", columns="m", values="v"); pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index].round(1)
    pv.columns = [f"M{m:02d}" for m in pv.columns]
    print(f"\nB. PIVOT: average ARRIVAL delay (minutes), airline x month, top 8 airlines by volume   (query time {time.time() - t:.2f} s)")
    show(pv.reset_index().fillna("-"))
    con.close()


# ------------------------------------------------------------------------------------------------ stage 2
def stage2():
    banner(f"STAGE 2 - LIVE ALGORITHM COMPARISON   ({TRAIN_N:,} train flights / {TEST_N:,} test flights, fixed seed {SEED})")
    df = features(); tr, va, te = split_masks(df); y = df[TARGET].to_numpy()
    feats = ALL_FEATURES; assert_no_leakage(feats, allow_rebuild=False)
    itr, ite, iva = stratified_idx(y, tr, TRAIN_N, SEED), stratified_idx(y, te, TEST_N, SEED), stratified_idx(y, va, VAL_N, SEED)
    ytr, yva, yte = y[itr], y[iva], y[ite]; cw = class_weights(ytr); w = np.where(ytr == 1, cw[1], cw[0]).astype("float32")
    d_tr, d_va, d_te = df.iloc[itr], df.iloc[iva], df.iloc[ite]
    X_tr, X_va, X_te = ordinal_codes(d_tr, feats), ordinal_codes(d_va, feats), ordinal_codes(d_te, feats)
    card = {c: len(df[c].cat.categories) for c in CAT_COLS}
    print(f"Task: will a flight arrive 15+ min late?  Prediction moment T-60 (1 h before departure), {len(feats)} features, balanced class weights.")
    print(f"Train: Jan-Aug sample | Test: Nov-Dec sample | Sep sample ({VAL_N:,}) is used only for early stopping and the F1 threshold (no tuning live).\n", flush=True)
    dt_depth = json.load(open(ROOT / "results/model_comparison/dt_best_depth.json"))["best_depth"]
    lgb_p = json.load(open(ROOT / "results/model_comparison/lgbm_best_params.json"))["best_params"]; xgb_p = json.load(open(ROOT / "results/model_comparison/xgb_best_params.json"))["best_params"]
    order = [("dummy_majority", "Majority dummy"), ("naive_bayes_gaussian", "Naive Bayes (Gaussian)"), ("naive_bayes_categorical", "Naive Bayes (Categorical)"),
             ("logistic_regression", "Logistic Regression"), ("decision_tree", "Decision Tree"), ("knn", "KNN (k=50)"), ("random_forest", "Random Forest"),
             ("xgboost", "XGBoost (tuned)"), ("lightgbm", "LightGBM")]
    res = []
    for i, (key, label) in enumerate(order, 1):
        print(f"[{i}/{len(order)}] training {label} ...", end=" ", flush=True); t = time.time()
        if key == "dummy_majority": sv, st = np.zeros(len(yva)), np.zeros(len(yte))
        elif key == "lightgbm":
            import lightgbm as lgb
            cat = [c for c in feats if c in CAT_COLS]; dtr = lgb.Dataset(d_tr[feats], ytr, weight=w, categorical_feature=cat, free_raw_data=False)
            dva = lgb.Dataset(d_va[feats], yva, categorical_feature=cat, reference=dtr)
            bst = lgb.train({**LGB_BASE, **lgb_p, "num_threads": NJOBS, "seed": SEED}, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            sv, st = bst.predict(d_va[feats], num_iteration=bst.best_iteration), bst.predict(d_te[feats], num_iteration=bst.best_iteration)
        elif key == "xgboost":
            m = make_xgb(xgb_p, SEED, False, NJOBS); m.fit(X_tr, ytr, sample_weight=w, eval_set=[(X_va, yva)], verbose=False)
            sv, st = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
        else:
            spec = sk_model_spec(key, SEED, NJOBS, dt_depth=dt_depth, card=card, feats=feats); pre = sk_pre(feats, spec["kind"]); m = spec["model"]
            cast = (lambda a: a.astype("int32")) if spec["kind"] == "catnb" else (lambda a: a)
            Xt = cast(pre.fit_transform(X_tr)); m.fit(Xt, ytr, sample_weight=w) if spec["weighted"] else m.fit(Xt, ytr)
            def prob(X):
                Z = cast(pre.transform(X)); step = 25_000 if key == "knn" else Z.shape[0]
                return np.concatenate([m.predict_proba(Z[j:j + step])[:, 1] for j in range(0, Z.shape[0], step)])
            sv, st = prob(X_va), prob(X_te)
        tt = time.time() - t
        if key == "dummy_majority": met = dict(roc_auc=0.5, pr_auc=float(yte.mean()), f1=0.0)
        else: thr, _ = best_f1_threshold(yva, sv); met = metrics_at(yte, st, thr)
        res.append(dict(model=label, pr_auc=met["pr_auc"], roc_auc=met["roc_auc"], f1=met["f1"], train_s=tt)); print(f"done in {tt:5.1f} s   PR-AUC {met['pr_auc']:.3f}", flush=True)
    R = pd.DataFrame(res).sort_values("pr_auc", ascending=False).reset_index(drop=True); R.insert(0, "rank", R.index + 1)
    base = float(yte.mean()); R["lift"] = R.pr_auc / base
    R.assign(train_sample=len(itr), test_sample=len(ite), test_base_rate=base, seed=SEED).to_csv(OUT / "live_comparison.csv", index=False)
    T = pd.DataFrame({"rank": R["rank"], "model": R.model, "PR-AUC": R.pr_auc.map("{:.3f}".format), "lift": R.lift.map("{:.2f}x".format), "ROC-AUC": R.roc_auc.map("{:.3f}".format),
                      "F1": R.f1.map("{:.3f}".format), "train time": R.train_s.map("{:.1f} s".format), "": ["<-- TOP" if i == 0 else "" for i in range(len(R))]})
    print("\nRESULT (test flights Nov-Dec, ranked by PR-AUC):"); show(T)
    if len(R) > 1 and R.pr_auc.iloc[0] - R.pr_auc.iloc[1] < 0.01:
        print(f"Note: ranks 1 and 2 differ by only {R.pr_auc.iloc[0] - R.pr_auc.iloc[1]:.3f} PR-AUC (< 0.01 counts as practically equivalent in the full-data study), so their order is not meaningful.")
    print(f"\nBase rate of the test sample: {100 * base:.1f}% of flights arrive 15+ min late  ->  a model that ignores every feature scores PR-AUC = {base:.3f}.")
    print("Full-data results (all Jan-Aug training rows): see results/model_comparison/comparison_table.csv")
    print(f"Table saved to results/demo/live_comparison.csv", flush=True)


# ------------------------------------------------------------------------------------------------ stage 3
def stage3():
    banner("STAGE 3 - LIVE PREDICTION   (saved full-data T-60 LightGBM model, 5 real flights from Nov-Dec)")
    import lightgbm as lgb
    mp, mj = ROOT / "models/lightgbm_t60.txt", ROOT / "models/lightgbm_t60.json"
    if not mp.exists() or not mj.exists(): raise FileNotFoundError("models/lightgbm_t60.txt missing - the demo never trains the full model; restore it from the project backup")
    bst = lgb.Booster(model_file=str(mp)); meta = json.load(open(mj)); thr = meta["threshold_sep_max_f1"]; feats = bst.feature_name()
    df = features(); tr, va, te = split_masks(df); y = df[TARGET].to_numpy()
    print(f"Model: LightGBM, {bst.num_trees()} trees, trained on all Jan-Aug flights ({meta['model'].split('(')[-1].split(')')[0] if '(' in meta['model'] else '3.84M flights'}); decision threshold {thr:.3f} (max-F1 on September).\n")
    rng = np.random.default_rng(SEED); pos = np.nonzero(te & (y == 1))[0]; neg = np.nonzero(te & (y == 0))[0]
    pick = list(rng.choice(pos, 3, replace=False)) + list(rng.choice(neg, 2, replace=False))
    sub = df.iloc[pick]; X = sub[feats]
    prob = bst.predict(X); contrib = bst.predict(X, pred_contrib=True)
    con = connect(read_only=True); keys = ",".join(str(int(k)) for k in sub.flight_key)
    act = con.execute(f"SELECT flight_key, arrival_delay_min FROM fact_flight WHERE flight_key IN ({keys})").df().set_index("flight_key").arrival_delay_min; con.close()
    ok = 0
    for n, (i, (_, r)) in enumerate(zip(range(len(pick)), sub.iterrows()), 1):
        d = str(int(r.date_key)); date = f"{d[:4]}-{d[4:6]}-{d[6:]}"; pred = prob[i] >= thr; a = float(act.loc[r.flight_key]); outcome = a >= 15; ok += int(pred == outcome)
        st = r.prev_leg_status.replace("_", " ")
        if r.prev_leg_status in ("landed", "airborne") and pd.notna(r.prev_leg_dep_delay): st += f" (left its gate {abs(r.prev_leg_dep_delay):.0f} min {'late' if r.prev_leg_dep_delay > 0 else 'early' if r.prev_leg_dep_delay < 0 else 'on time'})"
        print(f"FLIGHT {n}   {date}   {r.airline}   {r.origin} -> {r.dest}   scheduled departure {int(r.sched_dep_hour):02d}:xx")
        print(f"  Origin weather             : {r.o_wx_class}")
        print(f"  Inbound aircraft at T-60   : {st}")
        print(f"  Model says                 : P(delayed) = {prob[i]:.2f}  ->  predicted {'DELAYED' if pred else 'ON TIME'}   (threshold {thr:.2f})")
        print(f"  What actually happened     : arrived {abs(a):.0f} min {'late' if a > 0 else 'early' if a < 0 else 'on time'}  ->  {'DELAYED' if outcome else 'ON TIME'}     [{'CORRECT' if pred == outcome else 'MISSED'}]")
        top = np.argsort(-np.abs(contrib[i, :-1]))[:3]
        def val(f):
            v = r[f]; return "n/a" if pd.isna(v) else (str(v) if f in CAT_COLS else f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}")
        print("  Top 3 drivers (SHAP, +raises / -lowers delay risk): " + "; ".join(f"{feats[j]}={val(feats[j])} ({contrib[i, j]:+.2f})" for j in top) + "\n")
    print(f"Result: {ok} of {len(pick)} flights classified correctly at the September-tuned threshold (a 5-flight sample says nothing about accuracy; see Stage 2 / the report).", flush=True)


STAGES = {"1": ("Warehouse is real", stage1), "2": ("Live algorithm comparison", stage2), "3": ("Live prediction", stage3)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--stage", choices=["1", "2", "3", "all"], required=True)
    a = ap.parse_args(); logf = open(LOG, "w"); real = sys.stdout; sys.stdout = Tee(real, logf)
    run = list(STAGES) if a.stage == "all" else [a.stage]; times = {}; t_all = time.time()
    print(f"LIVE DEMO  -  Flight Delay & Airport Operations Intelligence   ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    for s in run:
        t = time.time()
        try: STAGES[s][1]()
        except Exception as e:                                              # clear message instead of a traceback; details go to a log file
            err = OUT / f"demo_error_stage{s}_{TS}.txt"; err.write_text(traceback.format_exc())
            print(f"\n!! Stage {s} ({STAGES[s][0]}) could not run: {type(e).__name__}: {str(e)[:200]}\n   Saved outputs from earlier runs are in results/demo/ (live_comparison.csv, demo_run_*.txt). Error details: {err.name}", flush=True)
        times[s] = time.time() - t
    if len(run) > 1 or True:
        banner("RUNTIME"); [print(f"  Stage {s}: {STAGES[s][0]:<28s} {times[s]:6.1f} s") for s in run]; print(f"  Total: {time.time() - t_all:.1f} s")
    print(f"\n(full output saved to results/demo/{LOG.name})"); sys.stdout = real; logf.close()
