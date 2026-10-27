"""Step 4 - explainability: LightGBM gain importance + SHAP (summary, dependence), an interaction check
(snow x northern airport x winter), subgroup performance vs baselines, and a readable depth-4/5 Decision Tree
with IF-THEN rules.  Needs the models/predictions produced by 04_train_compare.py (mining_cache/)."""
import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, _tree
from ml_common import *
from common import outdir, set_plot_style, SEED

plt = set_plot_style(); out = outdir("explain"); pd.set_option("display.width", 220)
MODEL = CACHE / "lgbm_t60_seed42.txt"; assert MODEL.exists(), "run 04_train_compare.py first"
bst = lgb.Booster(model_file=str(MODEL)); FEATS = bst.feature_name(); assert_no_leakage(FEATS)
df = load_features(); tr, va, te = split_masks(df); y = df[TARGET].to_numpy()
dte = df[te].reset_index(drop=True); yte = y[te]
CW = class_weights(y[tr])

# ---- 1. gain importance ----------------------------------------------------------------------------------------
imp = pd.DataFrame({"feature": FEATS, "gain": bst.feature_importance("gain"), "splits": bst.feature_importance("split")})
imp["gain_share_pct"] = (100 * imp.gain / imp.gain.sum()).round(2); imp = imp.sort_values("gain", ascending=False); imp.to_csv(out / "lgbm_gain_importance.csv", index=False)
print("Top features by gain:\n", imp.head(15).round(2).to_string(index=False))
top = imp.head(20).iloc[::-1]; fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top.feature, top.gain_share_pct, color="#2a7ab0")
ax.set_xlabel("Share of total split gain (%)"); ax.set_ylabel("Feature"); ax.set_title("LightGBM (T-60 model): top 20 features by gain"); fig.savefig(out / "lgbm_gain_importance.png"); plt.close(fig)
grp = {"Predecessor status/delay (T-60)": ["prev_leg_status", "prev_leg_arr_delay", "prev_leg_dep_delay"], "Schedule turn structure": ["prev_leg_sched_arr_minus_t60", "planned_turn_min", "leg_number_of_day"],
       "Historical delay rates": HIST, "Weather (origin+dest)": WX_NUM + WX_CAT, "Time of day / calendar": ["sched_dep_hour", "sched_arr_hour", "month", "day_of_week", "week_of_year", "weekend_flag", "holiday_flag", "season"],
       "Airline / airport / route": ["airline", "origin", "dest", "origin_state", "dest_state", "route"], "Distance / schedule": ["distance_miles", "distance_band", "scheduled_time_min", "interstate_flag"]}
G = pd.DataFrame([(k, imp[imp.feature.isin(v)].gain_share_pct.sum()) for k, v in grp.items()], columns=["feature_group", "gain_share_pct"]).sort_values("gain_share_pct", ascending=False).round(2)
G.to_csv(out / "lgbm_gain_by_group.csv", index=False); print("\nGain by group:\n", G.to_string(index=False))

# ---- 2. SHAP on a stratified-random 50k test sample (exact TreeSHAP via LightGBM pred_contrib; native categoricals) ----
import shap
rng = np.random.default_rng(SEED); ix = np.sort(rng.choice(len(dte), 50_000, replace=False)); S = dte.iloc[ix][FEATS]
contrib = bst.predict(S, pred_contrib=True); sv = contrib[:, :-1]; base = contrib[0, -1]
Xnum = S.copy()
for c in CAT_COLS:
    if c in Xnum: Xnum[c] = Xnum[c].cat.codes.astype("float32")
imp_shap = pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False); imp_shap.to_csv(out / "shap_mean_abs.csv", index=False)
print("\nTop mean |SHAP| (log-odds):\n", imp_shap.head(12).round(3).to_string(index=False))
shap.summary_plot(sv, Xnum, feature_names=FEATS, max_display=20, show=False); fig = plt.gcf(); fig.set_size_inches(10, 8); plt.title("SHAP summary: impact on log-odds of arriving 15+ min late (50k test flights)")
plt.xlabel("SHAP value (impact on model output, log-odds)"); fig.savefig(out / "shap_summary.png", bbox_inches="tight"); plt.close(fig)
for f_, inter in [("prev_leg_dep_delay", "sched_dep_hour"), ("sched_dep_hour", "prev_leg_status"), ("o_snow", "month"), ("hist_route_rate", "sched_dep_hour"), ("planned_turn_min", "prev_leg_dep_delay")]:
    if f_ not in FEATS: continue
    shap.dependence_plot(f_, sv, Xnum, feature_names=FEATS, interaction_index=inter if inter in FEATS else None, show=False)
    fig = plt.gcf(); fig.set_size_inches(8, 5.5); plt.title(f"SHAP dependence: {f_} (colour: {inter})"); plt.ylabel(f"SHAP value for {f_} (log-odds)"); fig.savefig(out / f"shap_dependence_{f_}.png", bbox_inches="tight"); plt.close(fig)

# ---- 3. nonlinear interaction check: snow x northern airport x winter ----------------------------------------------
NORTH = {"NY", "MA", "MN", "MI", "WI", "IL", "PA", "OH", "NJ", "CT", "VT", "NH", "ME", "RI", "IA", "ND", "SD", "NE", "CO", "MT", "WY", "ID", "UT", "IN", "AK"}
SOUTH = {"TX", "FL", "GA", "AZ", "CA", "LA", "AL", "MS", "SC", "NM", "HI"}
base_df = dte.copy(); res = []
cf0 = base_df.copy(); cf1 = base_df.copy(); cf0["o_snow"] = np.float32(0.0); cf1["o_snow"] = np.float32(30.0)      # counterfactual: no snow vs 30 mm snowfall at origin
for c in ["o_snwd"]: cf1[c] = np.float32(np.nan_to_num(cf1[c], nan=0) + 100)
d_ = bst.predict(cf1[FEATS]) - bst.predict(cf0[FEATS])
for lab, m in [("Northern origin, December (winter)", base_df.origin_state.isin(NORTH) & (base_df.month == 12)), ("Southern origin, December", base_df.origin_state.isin(SOUTH) & (base_df.month == 12)),
               ("Northern origin, November", base_df.origin_state.isin(NORTH) & (base_df.month == 11)), ("Southern origin, November", base_df.origin_state.isin(SOUTH) & (base_df.month == 11))]:
    mm = m.to_numpy(); res.append(dict(group=lab, flights=int(mm.sum()), actual_delay_rate_snow_gt0=yte[mm & (base_df.o_snow.to_numpy() > 0)].mean(), actual_delay_rate_no_snow=yte[mm & (base_df.o_snow.fillna(0).to_numpy() == 0)].mean(),
                    model_effect_of_30mm_snow_on_P_delay=d_[mm].mean()))
I = pd.DataFrame(res).round(4); I.to_csv(out / "interaction_snow_north_winter.csv", index=False); print("\nSnow x north x winter (counterfactual model effect, +30 mm snowfall & +100 mm snow depth):\n", I.to_string(index=False))

# ---- 4. subgroup performance of the main model vs baselines (shows native NaN / high-cardinality handling) ---------
from sklearn.metrics import roc_auc_score, average_precision_score
runs = {m: np.load(CACHE / "preds" / "t60" / f"{m}__42.npz")["test"] for m in ["lightgbm", "xgboost_tuned", "xgboost_native_tuned", "xgboost", "logistic_regression", "random_forest", "decision_tree"] if (CACHE / "preds" / "t60" / f"{m}__42.npz").exists()}
tr_route = df[tr].route.value_counts(); rare = dte.route.map(tr_route).fillna(0).to_numpy() < 100
subs = {"All test flights": np.ones(len(dte), bool), "Origin weather missing (SJU, STT, ... NaN weather)": dte.o_weather_missing.to_numpy() == 1, "Origin snow > 0": dte.o_snow.fillna(0).to_numpy() > 0,
        "Rare routes (<100 training flights)": rare, "Predecessor 'not_departed' at T-60": (dte.prev_leg_status == "not_departed").to_numpy(), "Predecessor 'landed' at T-60": (dte.prev_leg_status == "landed").to_numpy(),
        "No predecessor (first leg / broken chain)": (dte.prev_leg_status == "no_prev_leg").to_numpy()}
rows = []
for sname, m in subs.items():
    for mname, s in runs.items():
        if m.sum() < 500 or yte[m].min() == yte[m].max(): continue
        rows.append(dict(subgroup=sname, flights=int(m.sum()), delayed_rate=round(yte[m].mean(), 3), model=mname, pr_auc=round(average_precision_score(yte[m], s[m]), 4), roc_auc=round(roc_auc_score(yte[m], s[m]), 4)))
SG = pd.DataFrame(rows); SG.to_csv(out / "subgroup_performance.csv", index=False); print("\nSubgroup PR-AUC:\n", SG.pivot(index=["subgroup", "flights", "delayed_rate"], columns="model", values="pr_auc").to_string())

# ---- 5. readable Decision Tree (depth 4 and 5) -> IF-THEN rules -----------------------------------------------------
RD_NUM = [c for c in NUM_COLS if c not in ("hist_route_rate",) ] + ["hist_route_rate"]
RD_CAT = ["prev_leg_status", "season", "o_wx_class", "d_wx_class", "distance_band", "airline"]
def design(d):
    X = d[RD_NUM].astype("float32").copy()
    for c in RD_CAT:
        for lv in df[c].cat.categories: X[f"{c}={lv}"] = (d[c] == lv).astype("float32")
    return X
Xtr, Xva, Xte = design(df[tr]), design(df[va]), design(df[te]); imp_ = SimpleImputer(strategy="median").fit(Xtr)
cols = Xtr.columns; Xtr, Xva, Xte = [pd.DataFrame(imp_.transform(x), columns=cols) for x in (Xtr, Xva, Xte)]
w = np.where(y[tr] == 1, CW[1], CW[0]); lines = []; summ = []
ref = {m: np.load(CACHE / "preds" / "t60" / f"{m}__42.npz") for m in ["lightgbm"] if (CACHE / "preds" / "t60" / f"{m}__42.npz").exists()}
for depth in (4, 5):
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=2000, random_state=SEED).fit(Xtr, y[tr], sample_weight=w)
    sva, ste = dt.predict_proba(Xva)[:, 1], dt.predict_proba(Xte)[:, 1]; thr, _ = best_f1_threshold(y[va], sva); m = metrics_at(yte, ste, thr)
    summ.append(dict(model=f"Decision Tree depth {depth} (readable features)", **{k: round(v, 4) for k, v in m.items() if k in ("pr_auc", "roc_auc", "f1", "precision", "recall")}))
    if depth == 4 or depth == 5:
        t = dt.tree_; leaf_tr = dt.apply(Xtr); leaf_te = dt.apply(Xte); rate = pd.Series(y[tr]).groupby(leaf_tr).agg(["mean", "size"]); rate_te = pd.Series(yte).groupby(leaf_te).agg(["mean", "size"])
        def simplify(conds):
            """merge repeated bounds on one feature; drop conditions on imputed values (a predecessor that has not departed has no departure delay;
            one that has not landed has no arrival delay) - those thresholds only reflect median imputation."""
            txt = " ".join(conds); drop = set()
            if "prev_leg_status=not_departed" in txt: drop |= {"prev_leg_dep_delay", "prev_leg_arr_delay"}
            if "prev_leg_status=landed" not in txt or "NOT prev_leg_status=landed" in txt: drop |= {"prev_leg_arr_delay"}
            lo, hi, other = {}, {}, []
            for c in conds:
                if " <= " in c and "=" not in c.split(" <= ")[0]: f_, v = c.split(" <= "); hi[f_] = min(hi.get(f_, 1e18), float(v))
                elif " > " in c: f_, v = c.split(" > "); lo[f_] = max(lo.get(f_, -1e18), float(v))
                else: other.append(c)
            out = [c for c in other]
            for f_ in sorted(set(lo) | set(hi)):
                if f_ in drop: continue
                if f_ in lo and f_ in hi: out.append(f"{lo[f_]:.1f} < {f_} <= {hi[f_]:.1f}")
                elif f_ in lo: out.append(f"{f_} > {lo[f_]:.1f}")
                else: out.append(f"{f_} <= {hi[f_]:.1f}")
            return " AND ".join(out) if out else "(all flights)"
        def walk(node, conds):
            if t.feature[node] == _tree.TREE_UNDEFINED:
                r = rate.loc[node]; rt = rate_te.loc[node] if node in rate_te.index else None; wv = t.value[node][0]; pred = "DELAYED" if wv[1] / wv.sum() >= 0.5 else "on time"
                lines.append((depth, r["mean"], int(r["size"]), pred, simplify(conds), (rt["mean"] if rt is not None else np.nan), (int(rt["size"]) if rt is not None else 0))); return
            f, th = cols[t.feature[node]], t.threshold[node]
            if "=" in f and th == 0.5: walk(t.children_left[node], conds + [f"NOT {f}"]); walk(t.children_right[node], conds + [f])
            else: walk(t.children_left[node], conds + [f"{f} <= {th:.1f}"]); walk(t.children_right[node], conds + [f"{f} > {th:.1f}"])
        walk(0, [])
for depth in (4, 5):
    R = pd.DataFrame([l for l in lines if l[0] == depth], columns=["depth", "train_delay_rate", "train_flights", "balanced_prediction", "rule", "test_delay_rate", "test_flights"]).sort_values("train_delay_rate", ascending=False)
    R["IF-THEN"] = R.apply(lambda r: f"IF {r.rule} THEN {r.balanced_prediction}  [train: {100*r.train_delay_rate:.0f}% delayed, n={r.train_flights:,} | test Nov-Dec: {100*r.test_delay_rate:.0f}% delayed, n={r.test_flights:,}]", axis=1)
    R.drop(columns="depth").to_csv(out / f"decision_tree_depth{depth}_rules.csv", index=False)
    (out / f"decision_tree_depth{depth}_rules.txt").write_text("\n".join(R["IF-THEN"]))
    if depth == 4: print("\nDepth-4 rules (highest delay rate first):\n" + "\n".join(R["IF-THEN"].head(8)) + "\n...\n" + "\n".join(R["IF-THEN"].tail(3)))
if ref: s = ref["lightgbm"]; thr, _ = best_f1_threshold(y[va], s["val"]); m = metrics_at(yte, s["test"], thr); summ.append(dict(model="LightGBM (main)", **{k: round(v, 4) for k, v in m.items() if k in ("pr_auc", "roc_auc", "f1", "precision", "recall")}))
ST = pd.DataFrame(summ); ST.to_csv(out / "explainability_tradeoff.csv", index=False); print("\nExplainability trade-off (test):\n", ST.to_string(index=False))
from sklearn.tree import export_text
(out / "decision_tree_depth5_export_text.txt").write_text(export_text(dt, feature_names=list(cols), max_depth=5))
fig, ax = plt.subplots(figsize=(18, 8)); from sklearn.tree import plot_tree
dt4 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2000, random_state=SEED).fit(Xtr, y[tr], sample_weight=w)
plot_tree(dt4, feature_names=list(cols), class_names=["on time", "delayed"], filled=True, impurity=False, proportion=True, fontsize=7, ax=ax); ax.set_title("Decision tree, depth 4 (balanced class weights): readable rules for the report")
fig.savefig(out / "decision_tree_depth4.png"); plt.close(fig)

# ---- 6. fair prediction latency (model-only, on an already-prepared test matrix) -----------------------------------------
import time
lat = []
for rep in range(3):
    t0 = time.time(); bst.predict(dte[FEATS]); lat.append(time.time() - t0)
pd.DataFrame([dict(model="LightGBM (T-60 main, seed 42)", test_rows=len(dte), predict_seconds_median=float(np.median(lat)), per_1000_flights_ms=float(1000 * np.median(lat) / len(dte) * 1000))]).to_csv(out / "lightgbm_predict_latency.csv", index=False)
print(f"\nLightGBM model-only predict on {len(dte):,} test rows: median {np.median(lat):.1f}s over 3 runs (cached 'predict_time_s' in the comparison table also includes pandas slicing of the test frame).")
