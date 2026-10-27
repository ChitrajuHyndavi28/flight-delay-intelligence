"""Builds the extra data files for the local dashboard (dashboard/data/*.json) from the project's cached predictions and result tables."""
import json
import numpy as np, pandas as pd
from common import ROOT, connect
from ml_common import CACHE, TARGET, split_masks

R = ROOT / "results"; OUT = ROOT / "dashboard" / "data"; OUT.mkdir(parents=True, exist_ok=True)
def dump(name, df): df = df.replace({np.nan: None}); json.dump(df.to_dict("records"), open(OUT / f"{name}.json", "w")); print(name, len(df))
f2 = lambda p, r: 5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0.0

# ---- precision-recall / "flag the riskiest X%" curves from the cached test-set scores -----------------------------------
df = pd.read_parquet(CACHE / "features_t60.parquet", columns=["split", TARGET]); te = df.split.eq("test").to_numpy(); y = df[TARGET].to_numpy()[te].astype(int); n = len(y); base = y.mean()
models = {"LightGBM": "lightgbm__42", "XGBoost (tuned)": "xgboost_native_tuned__42", "Neural network": "mlp__42", "Ensemble": "ensemble_lgb_xgb_mlp__42"}
grid = sorted(set(list(range(1, 41)) + list(range(42, 101, 2))))
rows = []
for task, d in [("T-60", "t60"), ("Day-ahead", "dayahead")]:
    for name, f in models.items():
        s = np.load(CACHE / "preds" / d / f"{f}.npz")["test"]; o = np.argsort(-s, kind="stable"); ctp = np.cumsum(y[o])
        for g in grid:
            k = max(1, int(round(n * g / 100))); tp = ctp[k - 1]; fp = k - tp; fn = y.sum() - tp; tn = n - k - fn
            p, r = tp / k, tp / y.sum(); rows.append(dict(task=task, model=name, flag_pct=g, precision=p, recall=r, f2=f2(p, r), accuracy=(tp + tn) / n, lift=p / base))
for name, f in {"LightGBM": "lightgbm__42", "XGBoost (tuned)": "xgboost__42"}.items():           # post-departure (cached by 08_post_departure.py)
    s = np.load(CACHE / "preds" / "post_departure" / f"{f}.npz")["test"]; o = np.argsort(-s, kind="stable"); ctp = np.cumsum(y[o])
    for g in grid:
        k = max(1, int(round(n * g / 100))); tp = ctp[k - 1]; fn = y.sum() - tp; tn = n - k - fn; p, r = tp / k, tp / y.sum()
        rows.append(dict(task="Post-departure", model=name, flag_pct=g, precision=p, recall=r, f2=f2(p, r), accuracy=(tp + tn) / n, lift=p / base))
dump("pr_curves", pd.DataFrame(rows))

# ---- explainability ---------------------------------------------------------------------------------------------------------
NICE = {"prev_leg_dep_delay": "Previous leg: departure delay", "week_of_year": "Week of year", "d_prcp": "Destination precipitation", "prev_leg_status": "Previous leg status",
        "planned_turn_min": "Planned turnaround", "sched_dep_hour": "Scheduled departure hour", "hist_route_rate": "Route's historical delay rate", "o_snow": "Origin snowfall",
        "airline": "Airline", "origin": "Origin airport", "dest": "Destination airport", "route": "Route", "prev_leg_arr_delay": "Previous leg: arrival delay",
        "prev_leg_sched_arr_minus_t60": "Previous leg scheduled arrival vs T-60", "leg_number_of_day": "Leg number of the day", "o_prcp": "Origin precipitation",
        "d_snow": "Destination snowfall", "o_awnd": "Origin wind", "d_awnd": "Destination wind", "o_wx_class": "Origin weather class", "d_wx_class": "Destination weather class",
        "month": "Month", "day_of_week": "Day of week", "sched_arr_hour": "Scheduled arrival hour", "distance_miles": "Distance", "scheduled_time_min": "Scheduled flight time",
        "hist_airline_rate": "Airline's historical delay rate", "hist_origin_rate": "Origin's historical delay rate", "o_tmax": "Origin max temperature", "d_tmax": "Destination max temperature"}
sh = pd.read_csv(R / "explain" / "shap_mean_abs.csv").head(15); sh["label"] = sh.feature.map(lambda f: NICE.get(f, f)); dump("shap", sh)
dump("gain_groups", pd.read_csv(R / "explain" / "lgbm_gain_by_group.csv"))
sg = pd.read_csv(R / "explain" / "subgroup_performance.csv"); dump("subgroups", sg[sg.model == "lightgbm"].drop(columns="model"))

# ---- ablation ------------------------------------------------------------------------------------------------------------------
ab = pd.read_csv(R / "model_comparison" / "ablation_lightgbm.csv"); ab["pr_auc"] = ab.pr_auc.astype(str).str.split(" ").str[0].astype(float)
ab["roc_auc"] = ab.roc_auc.astype(str).str.split(" ").str[0].astype(float); dump("ablation", ab[["setting", "n_features", "pr_auc", "roc_auc", "recall", "precision"]])

# ---- time / airline / airport / propagation ----------------------------------------------------------------------------------------
dump("by_dow", pd.read_csv(R / "olap" / "by_day_of_week.csv"))
hm = pd.read_csv(R / "olap" / "airline_x_month_avg_arr_delay.csv").melt(id_vars="airline", var_name="month", value_name="avg_arr_delay_min"); hm["month"] = hm.month.astype(int); dump("airline_month", hm)
w = pd.read_csv(R / "olap" / "airports_top15_worst_dep_delay.csv"); w["group"] = "Worst 15"; b = pd.read_csv(R / "olap" / "airports_bottom15_best_dep_delay.csv"); b["group"] = "Best 15"; dump("airports_rank", pd.concat([w, b]))
bf = pd.read_csv(R / "propagation" / "3B_buffer_absorption.csv"); dump("buffer", bf[bf.incoming_delay != "< 15 (none)"][["planned_buffer_min", "incoming_delay", "flights", "pct_dep_delayed15", "pct_absorbed"]])
dump("cancel_reasons", pd.read_csv(R / "root_cause_weather" / "2b_cancel_reason_overall.csv"))
cm = pd.read_csv(R / "root_cause_weather" / "2a_cause_share_by_month.csv"); dump("cause_month", cm)

# ---- airports: map + weather sensitivity ---------------------------------------------------------------------------------------------------
ac = pd.read_csv(R / "clustering" / "airport_clusters.csv"); geo = connect().execute("SELECT airport_iata_code AS airport, latitude AS lat, longitude AS lon, city FROM dim_airport").df()
ac = ac.merge(geo, on="airport", how="left"); ac["cluster_name"] = ac.cluster.map({0: "Mid-size, punctual", 1: "Large hubs, congested", 2: "Smaller, weather-exposed"}); dump("airport_map", ac)


# ---- model tables for all three scenarios (F1-tuned threshold) -------------------------------------------------------------------------------
MC = R / "model_comparison"; mrows = []
for task, f, skip in [("T-60", "model_comparison_t60_numeric.csv", ""), ("Day-ahead", "model_comparison_dayahead_numeric.csv", "")]:
    d = pd.read_csv(MC / f); d = d[~d.model.str.contains("sample|dummy|Majority|default")]
    for r in d.itertuples(): mrows.append(dict(task=task, model=r.model, pr_auc=r.pr_auc, roc_auc=r.roc_auc, accuracy=r.accuracy, precision=r.precision, recall=r.recall, f1=r.f1, f2=f2(r.precision, r.recall), pr_lo=r.lo, pr_hi=r.hi))
for task, f in [("T-60", "nn_and_ensemble_t60.csv"), ("Day-ahead", "nn_and_ensemble_dayahead.csv")]:
    for r in pd.read_csv(MC / f).itertuples():
        if "reference" not in r.model: mrows.append(dict(task=task, model=r.model.replace(" seed 42", ""), pr_auc=r.pr_auc, roc_auc=r.roc_auc, accuracy=r.accuracy, precision=r.precision, recall=r.recall, f1=r.f1, f2=f2(r.precision, r.recall)))
for r in pd.read_csv(MC / "post_departure_scenario.csv").itertuples():
    mrows.append(dict(task="Post-departure", model="XGBoost (tuned)" if "XGB" in r.model else "LightGBM", pr_auc=r.pr_auc, roc_auc=r.roc_auc, accuracy=r.accuracy, precision=r.precision, recall=r.recall, f1=r.f1, f2=r.f2))
dump("models", pd.DataFrame(mrows))
w = pd.read_csv(MC / "tuned_weather_effect.csv"); ci = w.delta_pr_auc_ci.str.strip("[]").str.split(",")
w["ci_lo"] = ci.str[0].astype(float); w["ci_hi"] = ci.str[1].astype(float); dump("weather_effect", w[["task", "model", "pr_auc_off", "pr_auc_on", "delta_pr_auc", "ci_lo", "ci_hi", "delta_roc_auc"]])

# ---- base paper vs ours (paper numbers typed from its Table 4; ours from the result files) ---------------------------------------------------------
prow = []
for sc, v in {"Day-ahead": [("Paper XGBoost", 69.5, 68.2, 57.2), ("Paper Random Forest", 68.7, 67.2, 56.2), ("Paper Neural Network", 66.1, 65.8, 54.0)],
              "T-60": [("Paper XGBoost", 70.9, 68.5, 58.1), ("Paper Random Forest", 70.1, 67.2, 56.9), ("Paper Neural Network", 66.6, 66.5, 54.7)],
              "Post-departure": [("Paper XGBoost", 89.9, 83.4, 80.3), ("Paper Random Forest", 90.2, 81.7, 79.4), ("Paper Neural Network", 86.5, 85.3, 78.8)]}.items():
    for nm, a, rc, f in v: prow.append(dict(scenario=sc, system=nm, source="paper", threshold="Paper", accuracy=a, precision=None, recall=rc, f2=f, pr_auc=None, roc_auc=None))
def ours(sc, name, th, acc, prec, rec, pr, roc):
    prow.append(dict(scenario=sc, system=f"Ours {name}" + (" (F1 threshold)" if th == "F1-tuned" else " (recall matched)"), source="ours", threshold="F1-tuned" if th == "F1-tuned" else "Recall matched to paper",
                     accuracy=round(100 * acc, 2), precision=round(100 * prec, 2), recall=round(100 * rec, 2), f2=round(100 * f2(prec, rec), 2), pr_auc=round(pr, 4), roc_auc=round(roc, 4)))
keep = {"LightGBM (main)": "LightGBM", "XGBoost (native categoricals, tuned)": "XGBoost", "Random Forest": "Random Forest"}
for sc, f in [("T-60", "model_comparison_t60_numeric.csv"), ("Day-ahead", "model_comparison_dayahead_numeric.csv")]:
    d = pd.read_csv(MC / f)
    for k, nm in keep.items():
        r = d[d.model == k].iloc[0]; ours(sc, nm, "F1-tuned", r.accuracy, r.precision, r.recall, r.pr_auc, r.roc_auc)
for sc, f in [("T-60", "nn_and_ensemble_t60.csv"), ("Day-ahead", "nn_and_ensemble_dayahead.csv")]:
    for r in pd.read_csv(MC / f).itertuples():
        if r.model.startswith("Ensemble"): ours(sc, "Ensemble", "F1-tuned", r.accuracy, r.precision, r.recall, r.pr_auc, r.roc_auc); ours(sc, "Ensemble", "matched", r.acc_at_recall68, r.prec_at_recall68, r.rec_at_recall68, r.pr_auc, r.roc_auc)
        elif r.model.startswith("Neural"): ours(sc, "Neural network", "F1-tuned", r.accuracy, r.precision, r.recall, r.pr_auc, r.roc_auc)
        elif r.model.startswith("XGBoost"): ours(sc, "XGBoost", "matched", r.acc_at_recall68, r.prec_at_recall68, r.rec_at_recall68, r.pr_auc, r.roc_auc)
for r in pd.read_csv(MC / "post_departure_scenario.csv").itertuples():
    nm = "XGBoost" if "XGB" in r.model else "LightGBM"; ours("Post-departure", nm, "F1-tuned", r.accuracy, r.precision, r.recall, r.pr_auc, r.roc_auc); ours("Post-departure", nm, "matched", r.acc_at_recall83, r.prec_at_recall83, r.rec_at_recall83, r.pr_auc, r.roc_auc)
dump("paper_vs_ours", pd.DataFrame(prow))

# ---- tuning table (particle swarm) ----------------------------------------------------------------------------------------------------------------
pb = json.load(open(MC / "pso_best_params.json")); mb = json.load(open(MC / "mlp_pso_best.json")); fin = pd.read_csv(MC / "tuned_final_weather_on_off.csv"); fin = fin[fin.weather == "ON"]
nm = pd.DataFrame(mrows); bef = lambda t, m: float(nm[(nm.task == t) & (nm.model == m)].pr_auc.iloc[0])
fmtp = lambda p: " · ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in p.items())
trow = []
for key, (t, mdl, kind, base) in {"t60/xgb": ("T-60", "XGBoost", "XGBoost (PSO-tuned)", "XGBoost (tuned, ordinal codes)"), "t60/lgb": ("T-60", "LightGBM", "LightGBM (PSO-tuned)", "LightGBM (main)"),
                                  "dayahead/xgb": ("Day-ahead", "XGBoost", "XGBoost (PSO-tuned)", "XGBoost (tuned, ordinal codes)"), "dayahead/lgb": ("Day-ahead", "LightGBM", "LightGBM (PSO-tuned)", "LightGBM (main)")}.items():
    trow.append(dict(task=t, model=mdl, values=fmtp(pb[key]["params"]), pr_auc_tuned=float(fin[(fin.task == t) & (fin.model == kind)].pr_auc.iloc[0]), pr_auc_before=bef(t, base)))
for key, t, f in [("t60", "T-60", "nn_and_ensemble_t60_tuned.csv"), ("dayahead", "Day-ahead", "nn_and_ensemble_dayahead_tuned.csv")]:
    mt = pd.read_csv(MC / f); mt = mt[mt.model.str.startswith("Neural")].pr_auc.iloc[0]
    trow.append(dict(task=t, model="Neural network", values=fmtp(mb[key]["params"]), pr_auc_tuned=float(mt), pr_auc_before=bef(t, "Neural Network (MLP)")))
dump("tuning", pd.DataFrame(trow))

# ---- descriptive tables (OLAP / root cause / propagation / clustering / rules) --------------------------------------------------------------------------
m = pd.read_csv(R / "olap" / "by_month.csv"); m["month_name"] = pd.to_datetime(m.month, format="%m").dt.strftime("%b"); dump("by_month", m)
dump("by_hour", pd.read_csv(R / "olap" / "by_dep_hour.csv"))
dump("airlines", pd.read_csv(R / "olap" / "airline_ranking.csv")[["airline", "airline_name", "flights", "on_time_pct", "avg_arr_delay_min", "cancel_rate_pct"]])
dump("weather_class", pd.read_csv(R / "root_cause_weather" / "2c_raw_by_weather_class.csv"))
c = pd.read_csv(R / "root_cause_weather" / "2a_cause_share_overall.csv").iloc[0]
dump("cause_share", pd.DataFrame([dict(cause=n_, delay_minutes=c[k + "_min"], share_pct=c[k + "_share_pct"]) for n_, k in [("Late aircraft", "late_aircraft"), ("Airline", "airline"), ("Air system (NAS)", "air_system"), ("Weather", "weather"), ("Security", "security")]]))
dump("propagation", pd.read_csv(R / "propagation" / "3A_dep_delay_vs_incoming_delay.csv")[["incoming_delay_bin", "flights", "mean_incoming_delay", "mean_dep_delay", "pct_dep_delayed15"]])
cl = pd.read_csv(R / "clustering" / "cluster_profiles.csv"); cl["name"] = cl.cluster.map({0: "Mid-size, punctual", 1: "Large hubs, congested", 2: "Smaller, weather-exposed"}); dump("clusters", cl)
dump("rules", pd.read_csv(R / "association_rules" / "top_cancellation_rules.csv").head(15)[["antecedent", "antecedent_flights", "cancelled", "confidence", "lift", "main_reason"]])
