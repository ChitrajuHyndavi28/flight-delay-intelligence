"""Step 5b - Association rules for flight cancellations (O6). Outputs -> results/association_rules/.
Part A: exhaustive rule mining over ALL scheduled flights (non-October, known origin weather): antecedent = conjunction of 1-3 discrete
  conditions, consequent = CANCELLED (and each DOT cancellation reason). support / confidence / lift are exact (same definitions as Apriori);
  computed on the distinct item-combination table, so no sampling and no minimum-support approximation.
Part B: FP-Growth (mlxtend) on the cancelled flights only, with the cancellation reason as an item, to profile what cancelled flights look like."""
import itertools
import numpy as np, pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from common import connect, outdir, set_plot_style, wx_class_sql

out = outdir("association_rules"); plt = set_plot_style(); con = connect()
MIN_ANT = 2000          # antecedent must cover >= 2,000 flights
MIN_CANCEL = 100        # and >= 100 cancellations
WX = wx_class_sql("w"); WXD = wx_class_sql("wd")
agg = con.execute(f"""
SELECT d.month_name AS month, CASE WHEN d.weekend_flag THEN 'weekend' ELSE 'weekday' END AS daytype,
  CASE WHEN f.scheduled_departure_min//60 BETWEEN 5 AND 11 THEN 'morning' WHEN f.scheduled_departure_min//60 BETWEEN 12 AND 17 THEN 'afternoon'
       WHEN f.scheduled_departure_min//60 BETWEEN 18 AND 21 THEN 'evening' ELSE 'night' END AS dep_time,
  l.airline_iata_code AS airline, CASE WHEN {WX}='clear' THEN 'o_clear' WHEN {WX}='moderate' THEN 'o_moderate_wx' ELSE 'o_severe_wx' END AS o_wx,
  CASE WHEN coalesce(w.snowfall_mm,0) > 0 THEN 'o_snow' ELSE 'o_nosnow' END AS o_snow,
  CASE WHEN coalesce(w.avg_wind_ms,0) > 8.9 THEN 'o_high_wind' ELSE 'o_normal_wind' END AS o_wind,
  CASE WHEN w.tmax_c <= 0 THEN 'o_freezing' ELSE 'o_above_0C' END AS o_temp,
  CASE WHEN {WXD}='clear' THEN 'd_clear' WHEN {WXD}='moderate' THEN 'd_moderate_wx' WHEN {WXD}='severe' THEN 'd_severe_wx' ELSE 'd_unknown' END AS d_wx,
  r.distance_band AS dist_band,
  count(*) AS n, sum(f.cancelled_flag) AS n_cancel,
  sum((f.cancellation_reason='A')::int) AS n_A, sum((f.cancellation_reason='B')::int) AS n_B, sum((f.cancellation_reason='C')::int) AS n_C
FROM fact_flight f JOIN dim_date d USING(date_key) JOIN dim_airline l USING(airline_key) JOIN dim_route r USING(route_key)
JOIN dim_weather w ON w.weather_key = f.origin_weather_key JOIN dim_weather wd ON wd.weather_key = f.destination_weather_key
WHERE d.month_number <> 10 AND f.origin_airport_key <> 0 AND f.origin_weather_key <> 0
GROUP BY ALL""").df()
ITEM_COLS = ["month", "daytype", "dep_time", "airline", "o_wx", "o_snow", "o_wind", "o_temp", "d_wx", "dist_band"]
N, NC = int(agg.n.sum()), int(agg.n_cancel.sum()); base = NC / N
REASON = {"A": "airline/carrier", "B": "weather", "C": "national air system", "D": "security"}
print(f"{N:,} flights, {NC:,} cancelled (base rate {100 * base:.3f}%); {len(agg):,} distinct item combinations")
base_by = {k: float(agg[f"n_{k}"].sum()) / N for k in "ABC"}

rows = []
for size in (1, 2, 3):
    for cols in itertools.combinations(ITEM_COLS, size):
        g = agg.groupby(list(cols), observed=True)[["n", "n_cancel", "n_A", "n_B", "n_C"]].sum().reset_index()
        g = g[(g.n >= MIN_ANT) & (g.n_cancel >= MIN_CANCEL)]
        for r in g.itertuples(index=False):
            vals = r[:size]; n, nc = r[size], r[size + 1]
            rows.append(dict(antecedent=" & ".join(f"{c}={v}" if c in ("month", "airline", "dist_band") else str(v) for c, v in zip(cols, vals)), size=size,
                             antecedent_flights=n, cancelled=nc, support=nc / N, antecedent_support=n / N, confidence=nc / n, lift=(nc / n) / base,
                             conf_airline=r[size + 2] / n, conf_weather=r[size + 3] / n, conf_nas=r[size + 4] / n))
rules = pd.DataFrame(rows); print(f"{len(rules):,} candidate rules (antecedent >= {MIN_ANT} flights, >= {MIN_CANCEL} cancellations)")

# prune: keep a longer rule only if it is more confident (by >= 25% relative) than every sub-rule that it contains
rules["items"] = rules.antecedent.str.split(" & ").map(frozenset); conf = dict(zip(rules["items"], rules.confidence))
def beats_subrules(r):
    if r["size"] == 1: return True
    return all(r.confidence >= 1.25 * conf[frozenset(s)] for k in range(1, r["size"]) for s in itertools.combinations(r["items"], k) if frozenset(s) in conf)
rules["non_redundant"] = rules.apply(beats_subrules, axis=1)
rules = rules.sort_values("lift", ascending=False).drop(columns="items")
rules.to_csv(out / "cancellation_rules_all.csv", index=False)
top = rules[rules.non_redundant & (rules.lift >= 2)].head(40).copy()
top["main_reason"] = top[["conf_airline", "conf_weather", "conf_nas"]].idxmax(axis=1).map({"conf_airline": "airline", "conf_weather": "weather", "conf_nas": "NAS"})
top.round(5).to_csv(out / "top_cancellation_rules.csv", index=False)
print("\n--- top non-redundant rules by lift (antecedent -> CANCELLED) ---")
print(top.head(20)[["antecedent", "antecedent_flights", "cancelled", "confidence", "lift", "main_reason"]].round(4).to_string(index=False))

# rules per cancellation reason: best lift of P(reason | antecedent) / P(reason)
rr = []
for k, col in (("A", "conf_airline"), ("B", "conf_weather"), ("C", "conf_nas")):
    t = rules[rules.non_redundant & (rules[col] * rules.antecedent_flights >= MIN_CANCEL / 2)].copy(); t["reason"] = REASON[k]; t["reason_lift"] = t[col] / base_by[k]
    rr.append(t.sort_values("reason_lift", ascending=False).head(10)[["reason", "antecedent", "antecedent_flights", col, "reason_lift"]].rename(columns={col: "confidence"}))
rr = pd.concat(rr); rr.round(5).to_csv(out / "rules_by_reason.csv", index=False); print("\n--- top rules by cancellation reason ---"); print(rr.round(4).to_string(index=False))

# ---- Part B: FP-Growth on cancelled flights -------------------------------------------------------------------------------------
canc = agg[agg.n_cancel > 0]
reason_rows = []
for k in "ABC":
    sub = canc[canc[f"n_{k}"] > 0]; reason_rows.append(sub.assign(reason=f"cancelled_{REASON[k].split('/')[0].replace(' ', '_')}", cnt=sub[f"n_{k}"]))
cb = pd.concat(reason_rows, ignore_index=True); cb = cb.iloc[np.repeat(np.arange(len(cb)), cb.cnt.astype(int).to_numpy())].reset_index(drop=True)   # one transaction per cancelled flight (security cancellations, 22 flights, omitted)
oh = pd.concat([pd.get_dummies(cb[c], dtype=bool) for c in ITEM_COLS if c not in ("month", "airline")] + [pd.get_dummies(cb["reason"], dtype=bool),
                pd.get_dummies(cb["month"], dtype=bool), pd.get_dummies(cb["airline"], prefix="airline", dtype=bool)], axis=1)
print(f"\nFP-Growth on {len(oh):,} cancelled-flight transactions x {oh.shape[1]} items")
fi = fpgrowth(oh, min_support=0.08, use_colnames=True, max_len=4)
ar = association_rules(fi, metric="lift", min_threshold=1.2)
is_reason = lambda s: len(s) == 1 and next(iter(s)).startswith("cancelled_")
ar = ar[ar.consequents.map(is_reason)].sort_values(["lift", "confidence"], ascending=False)
ar["antecedents"] = ar.antecedents.map(lambda s: " & ".join(sorted(s))); ar["consequents"] = ar.consequents.map(lambda s: next(iter(s)))
ar[["antecedents", "consequents", "support", "confidence", "lift"]].head(30).round(4).to_csv(out / "fpgrowth_rules_among_cancelled.csv", index=False)
print(ar[["antecedents", "consequents", "support", "confidence", "lift"]].head(12).round(3).to_string(index=False))

# ---- plot ------------------------------------------------------------------------------------------------------------------
t = top.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6.5)); ax.barh(t.antecedent, 100 * t.confidence, color="#8e6bbf")
for y, (c, l) in enumerate(zip(t.confidence, t.lift)): ax.text(100 * c, y, f" lift {l:.1f}", va="center", fontsize=8)
ax.axvline(100 * base, color="k", ls="--", lw=1); ax.set_xlabel("P(cancelled | antecedent)  [%]  (dashed = overall rate)"); ax.set_title("Top non-redundant cancellation rules")
ax.tick_params(axis="y", labelsize=7); fig.savefig(out / "top_cancellation_rules.png"); plt.close(fig)
print("saved to", out)
