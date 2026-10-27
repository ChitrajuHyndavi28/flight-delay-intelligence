"""Step 2 - Root-cause & weather impact (O3). Outputs -> results/root_cause_weather/.

a) cause shares over flights with arrival_delay_min >= 15 (the only flights where DOT fills the cause columns;
   verified: causes sum exactly to arrival delay there)
b) cancellation reasons by month / airport
c) weather effect: raw gap, gap controlled within (origin airport, month), logistic regression ORs,
   sensitivity with the warehouse's own 4-level label, DOT weather-delay vs excess delay on severe days.
Scope: airport-level / weather work excludes October (unmapped airports) and unknown-weather airports.
"""
import numpy as np, pandas as pd
from common import connect, outdir, set_plot_style, wx_class_sql, SEED

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
con = connect(); out = outdir("root_cause_weather"); plt = set_plot_style()
CAUSES = ["airline", "late_aircraft", "air_system", "weather", "security"]
CCOL = {"airline": "airline_delay_min", "late_aircraft": "late_aircraft_delay_min",
        "air_system": "air_system_delay_min", "weather": "weather_delay_min", "security": "security_delay_min"}
CLAB = {"airline": "Airline", "late_aircraft": "Late aircraft", "air_system": "Air system (NAS)",
        "weather": "Weather", "security": "Security"}
CCOLORS = {"airline": "#2a7ab0", "late_aircraft": "#d9822b", "air_system": "#6aa84f", "weather": "#8e6bbf", "security": "#999999"}
q = lambda s: con.execute(s).df()
def save(df, name, show=True):
    df.to_csv(out / f"{name}.csv", index=False)
    if show: print(f"\n--- {name} ---"); print(df.to_string(index=False))

# =============================== 2a. cause attribution ======================================
SUMS = ", ".join(f"sum({CCOL[c]}) AS {c}_min" for c in CAUSES)
FLOWN_DELAYED = "f.cancelled_flag=0 AND f.diverted_flag=0 AND f.arrival_delay_min>=15"

def shares(df):
    m = df[[f"{c}_min" for c in CAUSES]]
    for c in CAUSES: df[f"{c}_share_pct"] = (100 * m[f"{c}_min"] / m.sum(axis=1)).round(2)
    return df

print("=== 2a. Delay-cause shares (flights with arrival delay >= 15 min) ===")
ov = shares(q(f"SELECT count(*) AS delayed_flights, {SUMS} FROM fact_flight f WHERE {FLOWN_DELAYED}"))
save(ov, "2a_cause_share_overall")
byal = shares(q(f"SELECT l.airline_iata_code AS airline, count(*) AS delayed_flights, {SUMS} FROM fact_flight f "
                f"JOIN dim_airline l USING(airline_key) WHERE {FLOWN_DELAYED} GROUP BY 1"))
byal = byal.sort_values("late_aircraft_share_pct", ascending=False); save(byal.drop(columns=[f"{c}_min" for c in CAUSES]), "2a_cause_share_by_airline")
bymo = shares(q(f"SELECT d.month_number AS month, count(*) AS delayed_flights, {SUMS} FROM fact_flight f "
                f"JOIN dim_date d USING(date_key) WHERE {FLOWN_DELAYED} GROUP BY 1 ORDER BY 1"))
save(bymo.drop(columns=[f"{c}_min" for c in CAUSES]), "2a_cause_share_by_month")
top20 = q("""SELECT a.airport_iata_code AS airport FROM fact_flight f JOIN dim_airport a ON a.airport_key=f.origin_airport_key
  JOIN dim_date d USING(date_key) WHERE f.origin_airport_key<>0 AND d.month_number<>10 AND f.cancelled_flag=0 AND f.diverted_flag=0
  GROUP BY 1 ORDER BY count(*) DESC LIMIT 20""").airport.tolist()
byap = shares(q(f"""SELECT a.airport_iata_code AS airport, count(*) AS delayed_flights, {SUMS} FROM fact_flight f
  JOIN dim_airport a ON a.airport_key=f.origin_airport_key JOIN dim_date d USING(date_key)
  WHERE {FLOWN_DELAYED} AND f.origin_airport_key<>0 AND d.month_number<>10 AND a.airport_iata_code IN ({','.join(repr(a) for a in top20)})
  GROUP BY 1"""))
byap = byap.sort_values("weather_share_pct", ascending=False); save(byap.drop(columns=[f"{c}_min" for c in CAUSES]), "2a_cause_share_by_top20_airport")

def stacked(df, key, title, fname, horizontal=False, figsize=(12, 5.5), xl=None):
    fig, ax = plt.subplots(figsize=figsize); bottom = np.zeros(len(df))
    for c in CAUSES:
        v = df[f"{c}_share_pct"].to_numpy()
        (ax.barh if horizontal else ax.bar)(df[key].astype(str), v, **({"left": bottom} if horizontal else {"bottom": bottom}),
                                            label=CLAB[c], color=CCOLORS[c]); bottom += v
    lab = "Share of delay minutes (%) — delayed flights, arrival delay >= 15 min"
    if horizontal: ax.set_xlabel(lab); ax.set_ylabel(xl or key); ax.invert_yaxis()
    else: ax.set_ylabel(lab); ax.set_xlabel(xl or key)
    ax.set_title(title); ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13 if not horizontal else -0.1), frameon=False)
    fig.savefig(out / fname); plt.close(fig)
stacked(ov.assign(scope="All 2015 flights"), "scope", "Where delay minutes come from (2015, all months)", "2a_cause_share_overall.png", figsize=(6, 5.5), xl="")
stacked(byal, "airline", "Delay-cause mix by airline", "2a_cause_share_by_airline.png", xl="Airline")
stacked(bymo, "month", "Delay-cause mix by month", "2a_cause_share_by_month.png", xl="Month (2015)")
stacked(byap, "airport", "Delay-cause mix at the 20 busiest origin airports (Oct excluded)", "2a_cause_share_by_top20_airport.png", horizontal=True, figsize=(11, 8), xl="Origin airport")

# =============================== 2b. cancellation reasons ===================================
print("\n=== 2b. Cancellation reasons ===")
RN = {"A": "Airline", "B": "Weather", "C": "NAS", "D": "Security"}
tot = q("SELECT cancellation_reason AS reason, count(*) AS cancelled FROM fact_flight WHERE cancelled_flag=1 GROUP BY 1 ORDER BY 1")
tot["reason_name"] = tot.reason.map(RN); tot["share_pct"] = (100 * tot.cancelled / tot.cancelled.sum()).round(2); save(tot, "2b_cancel_reason_overall")
rec = q("""SELECT 'all 12 months (headline definition)' AS scope, count(*) AS flights, sum(cancelled_flag) AS cancelled,
  sum((cancellation_reason='B')::int) AS weather_coded FROM fact_flight
  UNION ALL SELECT 'excl. October (airport-level analyses)', count(*), sum(cancelled_flag), sum((cancellation_reason='B')::int)
  FROM fact_flight f JOIN dim_date d USING(date_key) WHERE d.month_number<>10
  UNION ALL SELECT 'excl. Oct & unknown-weather (Step 2c scope)', count(*), sum(cancelled_flag), sum((cancellation_reason='B')::int)
  FROM fact_flight f JOIN dim_date d USING(date_key) WHERE d.month_number<>10 AND f.origin_weather_key<>0 AND f.origin_airport_key<>0""")
rec["cancel_rate_pct"] = (100 * rec.cancelled / rec.flights).round(3); rec["weather_share_of_cancellations_pct"] = (100 * rec.weather_coded / rec.cancelled).round(2)
save(rec, "2b_cancellation_share_reconciliation")
cm = q("""SELECT d.month_number AS month, cancellation_reason AS reason, count(*) AS cancelled FROM fact_flight f JOIN dim_date d USING(date_key)
  WHERE cancelled_flag=1 GROUP BY 1,2""").pivot(index="month", columns="reason", values="cancelled").fillna(0).astype(int)
cm.columns = [RN[c] for c in cm.columns]; cm["total"] = cm.sum(axis=1); save(cm.reset_index(), "2b_cancel_reason_by_month")
ca = q("""SELECT a.airport_iata_code AS airport, cancellation_reason AS reason, count(*) AS cancelled FROM fact_flight f
  JOIN dim_airport a ON a.airport_key=f.origin_airport_key JOIN dim_date d USING(date_key)
  WHERE cancelled_flag=1 AND f.origin_airport_key<>0 AND d.month_number<>10 GROUP BY 1,2""").pivot(index="airport", columns="reason", values="cancelled").fillna(0)
ca.columns = [RN[c] for c in ca.columns]; ca["total"] = ca.sum(axis=1); ca = ca.sort_values("total", ascending=False).head(20).astype(int)
ca["weather_share_pct"] = (100 * ca["Weather"] / ca.total).round(1); save(ca.reset_index(), "2b_cancel_reason_by_top20_airport")
RC = {"Airline": "#2a7ab0", "Weather": "#8e6bbf", "NAS": "#6aa84f", "Security": "#999999"}
fig, axs = plt.subplots(1, 2, figsize=(17, 6.5))
cmm = cm.drop(columns="total"); bot = np.zeros(len(cmm))
for c in cmm.columns: axs[0].bar(cmm.index.astype(str), cmm[c], bottom=bot, label=c, color=RC[c]); bot += cmm[c].to_numpy()
axs[0].set_xlabel("Month (2015)"); axs[0].set_ylabel("Cancelled flights"); axs[0].set_title("By month (all flights)"); axs[0].legend(frameon=False)
cam = ca.drop(columns=["total", "weather_share_pct"]); bot = np.zeros(len(cam))
for c in cam.columns: axs[1].barh(cam.index, cam[c], left=bot, label=c, color=RC[c]); bot += cam[c].to_numpy()
axs[1].invert_yaxis(); axs[1].set_xlabel("Cancelled flights"); axs[1].set_ylabel("Origin airport"); axs[1].set_title("Top 20 airports by cancellations (Oct excluded)")
fig.suptitle("Cancellation reasons (DOT codes: A airline, B weather, C NAS, D security)", fontsize=15); fig.savefig(out / "2b_cancel_reasons.png"); plt.close(fig)

# =============================== 2c. weather effect =========================================
print("\n=== 2c. Weather effect (October & unknown-weather excluded) ===")
cells = q(f"""SELECT f.origin_airport_key AS ap, d.month_number AS month, {wx_class_sql('w')} AS wx, w.weather_severity AS whl,
  count(*) AS nall, sum(f.cancelled_flag) AS ncanc, sum(f.diverted_flag) AS ndiv, sum((f.cancelled_flag=1 AND f.cancellation_reason='B')::int) AS ncanc_b,
  sum(flown) AS n, sum(flown*(f.arrival_delay_min>=15)::int) AS nd,
  sum(CASE WHEN flown=1 THEN f.arrival_delay_min END) AS s_arr,
  {', '.join(f"sum(coalesce({CCOL[c]},0)) AS s_{c}" for c in CAUSES)}
  FROM (SELECT *, (cancelled_flag=0 AND diverted_flag=0)::int AS flown FROM fact_flight) f
  JOIN dim_date d USING(date_key) JOIN dim_weather w ON w.weather_key=f.origin_weather_key
  WHERE f.origin_airport_key<>0 AND f.origin_weather_key<>0 AND d.month_number<>10
  GROUP BY 1,2,3,4""")
cells["nbad"] = cells.nd + cells.ncanc + cells.ndiv   # disrupted = delayed>=15 OR cancelled OR diverted
print(f"{len(cells):,} (airport, month, class) cells; flights = {cells.nall.sum():,}")

METRICS = {"delay_rate_pct": ("nd", "n", 100), "avg_arr_delay_min": ("s_arr", "n", 1), "cancel_rate_pct": ("ncanc", "nall", 100),
           "disrupted_rate_pct": ("nbad", "nall", 100)}

def pooled(df, cls_col, cls):
    s = df[df[cls_col] == cls].sum(numeric_only=True)
    return {m: v * s[a] / s[b] for m, (a, b, v) in METRICS.items()}, s

def gaps(df, key, treat, ref="clear", min_n=50, boot=1000):
    """raw gap (pooled over all data) and gap controlled within (airport, month), weighted by treated-class flights.
    Cluster bootstrap over airports (1,000 resamples) for 95% CIs."""
    g = df.groupby(["ap", "month", key], as_index=False).sum(numeric_only=True)
    t = g[g[key] == treat].set_index(["ap", "month"]); r = g[g[key] == ref].set_index(["ap", "month"])
    j = t.join(r, lsuffix="_t", rsuffix="_r", how="inner")
    j = j[(j.n_t >= min_n) & (j.n_r >= min_n)].reset_index()
    rows = {}
    ap_all = np.sort(df.ap.unique()); idx = {a: i for i, a in enumerate(ap_all)}
    rng = np.random.default_rng(SEED)
    mult = np.stack([np.bincount(rng.integers(0, len(ap_all), len(ap_all)), minlength=len(ap_all)) for _ in range(boot)])
    # per-airport pooled sums for raw gap (all data, class-level)
    def apsum(cls):
        s = g[g[key] == cls].groupby("ap").sum(numeric_only=True).reindex(ap_all).fillna(0); return s
    st, sr = apsum(treat), apsum(ref)
    for m, (a, b, scale) in METRICS.items():
        raw = lambda w: scale * ((w @ st[a].to_numpy()) / (w @ st[b].to_numpy()) - (w @ sr[a].to_numpy()) / (w @ sr[b].to_numpy()))
        raw_pt = raw(np.ones(len(ap_all))); raw_bs = np.array([raw(w) for w in mult])
        d = scale * (j[f"{a}_t"] / j[f"{b}_t"] - j[f"{a}_r"] / j[f"{b}_r"]).to_numpy(); w0 = j[f"{b}_t"].to_numpy()
        ai = j.ap.map(idx).to_numpy()
        ctl = lambda mu: (mu[ai] * w0 * d).sum() / (mu[ai] * w0).sum()
        ctl_pt = ctl(np.ones(len(ap_all))); ctl_bs = np.array([ctl(w.astype(float)) for w in mult])
        pc = scale * st[a].sum() / st[b].sum(); pr = scale * sr[a].sum() / sr[b].sum()
        rows[m] = dict(scheme=key, treat=treat, ref=ref, metric=m, treated_rate=round(pc, 3), ref_rate=round(pr, 3),
                       raw_gap=round(raw_pt, 3), raw_lo=round(np.percentile(raw_bs, 2.5), 3), raw_hi=round(np.percentile(raw_bs, 97.5), 3),
                       controlled_gap=round(ctl_pt, 3), ctl_lo=round(np.percentile(ctl_bs, 2.5), 3), ctl_hi=round(np.percentile(ctl_bs, 97.5), 3),
                       matched_cells=len(j), matched_treated_flights=int(j.n_t.sum()), airports=len(ap_all))
    return pd.DataFrame(rows.values()), j

print("\nRaw flight-weighted class summary (derived classes):")
rs = pd.DataFrame({c: pooled(cells, "wx", c)[0] | {"flights": int(pooled(cells, "wx", c)[1].nall)} for c in ["clear", "moderate", "severe"]}).T
save(rs.reset_index().rename(columns={"index": "wx_class"}).round(3), "2c_raw_by_weather_class")

G = []
for key, treats, ref in [("wx", ["moderate", "severe"], "clear"), ("whl", ["Light", "Moderate", "Severe"], "Clear")]:
    for tr in treats:
        gp, _ = gaps(cells, key, tr, ref); G.append(gp)
G = pd.concat(G, ignore_index=True)
save(G[G.scheme == "wx"].drop(columns=["scheme", "airports"]), "2c_raw_vs_controlled_gaps_derived")
save(G[G.scheme == "whl"].drop(columns=["scheme", "airports"]), "2c_sensitivity_warehouse_label_gaps")
G.to_csv(out / "2c_all_gaps.csv", index=False)

# ---- coverage of the controlled comparison -------------------------------------------------------
def coverage(treat, key="wx", ref="clear", min_n=50):
    g = cells.groupby(["ap", "month", key], as_index=False).sum(numeric_only=True)
    t = g[g[key] == treat].set_index(["ap", "month"]); r = g[g[key] == ref].set_index(["ap", "month"])
    j = t.join(r[["n"]], rsuffix="_ref", how="left").reset_index()
    j["matched"] = (j.n >= min_n) & (j.n_ref.fillna(0) >= min_n)
    j["why_unmatched"] = np.where(j.matched, "", np.where(j.n < min_n, f"{treat} cell < {min_n} flights", f"no/thin clear cell (< {min_n}) in same airport-month"))
    return j
codes = dict(q("SELECT airport_key, airport_iata_code FROM dim_airport").values)
cov_rows, unm = [], {}
for tr in ["moderate", "severe"]:
    j_ = coverage(tr); m = j_[j_.matched]; u = j_[~j_.matched]
    cov_rows.append(dict(class_=tr, flown_flights=int(j_.n.sum()), matched_flights=int(m.n.sum()), matched_pct=round(100 * m.n.sum() / j_.n.sum(), 1),
                         cells=len(j_), matched_cells=len(m), unmatched_flights=int(u.n.sum()),
                         unmatched_small_cell=int(u[u.why_unmatched.str.contains("cell <")].n.sum()), unmatched_no_clear=int(u[~u.why_unmatched.str.contains("cell <")].n.sum())))
    ua = u.groupby("ap").n.sum().sort_values(ascending=False).head(15).reset_index(); ua["airport"] = ua.ap.map(codes)
    ua["share_of_unmatched_pct"] = (100 * ua.n / u.n.sum()).round(1); unm[f"{tr}_airports"] = ua[["airport", "n", "share_of_unmatched_pct"]].rename(columns={"n": "unmatched_flights"})
    um = u.groupby("month").n.sum().reset_index().rename(columns={"n": "unmatched_flights"}); um["share_of_unmatched_pct"] = (100 * um.unmatched_flights / u.n.sum()).round(1); unm[f"{tr}_months"] = um
cov_df = pd.DataFrame(cov_rows).rename(columns={"class_": "wx_class"}); save(cov_df, "2c_coverage_matched_cells")
for k, v in unm.items(): save(v, f"2c_unmatched_{k}")

# chart: raw vs controlled gaps (derived classes)
fig, axs = plt.subplots(1, 4, figsize=(21, 5))
Gd = G[G.scheme == "wx"]
for ax, m, lab in zip(axs, METRICS, ["Delay rate (pp, flown, arr >= 15)", "Avg arrival delay (min)", "Cancellation rate (pp)", "Disrupted rate (pp): delayed OR cancelled OR diverted"]):
    x = np.arange(2); wd = .35
    for k, (kind, col, lo, hi, c) in enumerate([("Raw gap", "raw_gap", "raw_lo", "raw_hi", "#b0b0b0"), ("Controlled gap (within airport x month)", "controlled_gap", "ctl_lo", "ctl_hi", "#2a7ab0")]):
        d = Gd[Gd.metric == m].set_index("treat").loc[["moderate", "severe"]]
        bc = ax.bar(x + (k - .5) * wd, d[col], wd, yerr=[d[col] - d[lo], d[hi] - d[col]], capsize=4, label=kind, color=c)
        for xi, v, h in zip(x + (k - .5) * wd, d[col], d[hi]): ax.annotate(f"{v:.1f}", (xi, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_ylim(0, Gd[Gd.metric == m][["raw_hi", "ctl_hi"]].max().max() * 1.15)
    ax.set_xticks(x); ax.set_xticklabels(["Moderate vs clear", "Severe vs clear"]); ax.set_ylabel(lab.split(":")[0]); ax.set_title(lab, fontsize=11); ax.axhline(0, color="k", lw=.8)
axs[0].legend(frameon=False, fontsize=9)
fig.suptitle("Weather effect at origin: raw vs controlled gap (95% CI, airport-cluster bootstrap)", fontsize=15); fig.savefig(out / "2c_raw_vs_controlled.png"); plt.close(fig)

# ---- logistic regression on 1M sample --------------------------------------------------------
import statsmodels.formula.api as smf, statsmodels.api as sm
def logit_or(target_sql, where, name):
    df = q(f"""SELECT * FROM (SELECT {target_sql} AS y, {wx_class_sql('w')} AS wx, a.airport_iata_code AS origin, d.month_number AS month,
        f.scheduled_departure_min//60 AS dep_hour, l.airline_iata_code AS airline, f.origin_weather_key AS wk
        FROM fact_flight f JOIN dim_date d USING(date_key) JOIN dim_weather w ON w.weather_key=f.origin_weather_key
        JOIN dim_airport a ON a.airport_key=f.origin_airport_key JOIN dim_airline l USING(airline_key)
        WHERE f.origin_airport_key<>0 AND f.origin_weather_key<>0 AND d.month_number<>10 AND {where})
        USING SAMPLE reservoir(1000000 ROWS) REPEATABLE ({SEED})""")
    df["wx"] = pd.Categorical(df.wx, categories=["clear", "moderate", "severe"])
    for c in ["origin", "airline"]: df[c] = df[c].astype("category")
    mod = smf.glm("y ~ C(wx, Treatment('clear')) + C(origin) + C(month) + C(dep_hour) + C(airline)", df,
                  family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.wk)[0]})
    res = []
    for k in mod.params.index:
        if "wx" in k:
            lo, hi = mod.conf_int().loc[k]
            res.append(dict(model=name, term=k.split("[T.")[1].rstrip("]"), odds_ratio=np.exp(mod.params[k]), ci95_lo=np.exp(lo), ci95_hi=np.exp(hi),
                            p_value=mod.pvalues[k], n=len(df), events=int(df.y.sum()), base_rate_pct=100 * df.y.mean()))
    return pd.DataFrame(res)
print("\nFitting logistic regressions (1M-row sample, cluster-robust SE by airport-day)...")
OR = pd.concat([logit_or("(f.arrival_delay_min>=15)::int", "f.cancelled_flag=0 AND f.diverted_flag=0", "delayed (arr>=15) ~ weather + airport + month + hour + airline"),
                logit_or("f.cancelled_flag", "TRUE", "cancelled ~ weather + airport + month + hour + airline")], ignore_index=True)
save(OR.round(4), "2c_logistic_odds_ratios")
fig, ax = plt.subplots(figsize=(9, 4.2)); yl = []
for i, (_, r) in enumerate(OR.iterrows()):
    ax.errorbar(r.odds_ratio, i, xerr=[[r.odds_ratio - r.ci95_lo], [r.ci95_hi - r.odds_ratio]], fmt="o", color="#2a7ab0", capsize=4)
    ax.annotate(f"{r.odds_ratio:.2f} [{r.ci95_lo:.2f}, {r.ci95_hi:.2f}]", (r.odds_ratio, i), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    yl.append(f"{r.term} vs clear — {'delayed' if r.model.startswith('delayed') else 'cancelled'}")
ax.axvline(1, color="k", lw=.8); ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl); ax.set_xlabel("Adjusted odds ratio (95% CI, airport-day clustered)")
ax.set_title("Weather odds ratios, adjusted for airport, month, hour, airline"); fig.savefig(out / "2c_logistic_odds_ratios.png"); plt.close(fig)

# ---- DOT weather-delay vs excess delay on severe days ----------------------------------------
print("\n=== 2c. DOT 'weather' delay vs total excess delay on severe days ===")
_, j = gaps(cells, "wx", "severe", boot=2)
w = j.n_t.to_numpy(); nS = w.sum()
def ctl_per_flight(num):  # controlled diff of per-flight mean, severe - clear, weight = severe flights
    d = j[f"{num}_t"] / j.n_t - j[f"{num}_r"] / j.n_r; return float((w * d).sum() / nS)
x_arr = ctl_per_flight("s_arr")
rows = [dict(component="Excess arrival delay, all flown flights (severe - clear)", min_per_flight=x_arr)]
for c in CAUSES: rows.append(dict(component=f"DOT cause: {CLAB[c]} (excess, severe - clear)", min_per_flight=ctl_per_flight(f"s_{c}")))
D = pd.DataFrame(rows); sum_c = D.min_per_flight[1:].sum()
D = pd.concat([D, pd.DataFrame([dict(component="Sum of 5 DOT causes (delays >= 15 min only)", min_per_flight=sum_c),
               dict(component="Unattributed (delays < 15 min, no cause recorded)", min_per_flight=x_arr - sum_c)])], ignore_index=True)
D["pct_of_excess"] = (100 * D.min_per_flight / x_arr).round(1); D["total_excess_minutes"] = (D.min_per_flight * nS).round(0)
D["severe_flights_in_matched_cells"] = int(nS); save(D.round(3), "2c_dot_weather_vs_excess_delay")
wx_dot = D.loc[D.component.str.contains("Weather"), "min_per_flight"].iloc[0]
nas = D.loc[D.component.str.contains("NAS"), "min_per_flight"].iloc[0]; late = D.loc[D.component.str.contains("Late aircraft"), "min_per_flight"].iloc[0]
print(f"\nExcess arrival delay on severe days: {x_arr:.2f} min/flight. DOT 'weather' accounts for {wx_dot:.2f} ({100*wx_dot/x_arr:.0f}%); "
      f"the other {x_arr-wx_dot:.2f} min ({100*(x_arr-wx_dot)/x_arr:.0f}%) is coded NAS {nas:.2f}, late-aircraft {late:.2f}, airline, or unattributed (<15 min). "
      f"Only the NAS part is plausibly miscoded weather; late-aircraft is secondary propagation. Weather-coded cancellations are separate (see 2b).")
sev = cells[cells.wx == "severe"].sum(numeric_only=True); clr = cells[cells.wx == "clear"].sum(numeric_only=True)
mix = pd.DataFrame({CLAB[c]: [100 * clr[f"s_{c}"] / sum(clr[f"s_{k}"] for k in CAUSES), 100 * sev[f"s_{c}"] / sum(sev[f"s_{k}"] for k in CAUSES)] for c in CAUSES},
                   index=["clear days", "severe days"]).round(2).reset_index().rename(columns={"index": "origin_weather"})
save(mix, "2c_cause_mix_clear_vs_severe")
wc = pd.DataFrame({"wx_class": ["clear", "moderate", "severe"], **{k: [cells[cells.wx == c][k].sum() for c in ["clear", "moderate", "severe"]] for k in ["ncanc", "ncanc_b", "nall"]}})
wc["weather_cancel_rate_pct"] = (100 * wc.ncanc_b / wc.nall).round(3); wc["share_of_cancels_coded_weather_pct"] = (100 * wc.ncanc_b / wc.ncanc).round(1)
save(wc, "2c_weather_coded_cancellations_by_class")

fig, ax = plt.subplots(figsize=(10, 5))
Dp = D.iloc[[0, 1, 2, 3, 4, 5, 7]].copy()
cols = ["#555555"] + [CCOLORS[c] for c in CAUSES] + ["#cccccc"]
ax.bar(range(len(Dp)), Dp.min_per_flight, color=cols); ax.axhline(0, color="k", lw=.8)
ax.set_xticks(range(len(Dp))); ax.set_xticklabels(["Total excess\narrival delay", *[CLAB[c] for c in CAUSES], "Unattributed\n(<15 min)"], fontsize=9)
ax.bar_label(ax.containers[0], fmt="%.2f", padding=2); ax.set_ylabel("Excess minutes per flight (severe - clear, within airport x month)")
ax.set_title("Severe-weather days: DOT 'weather' covers only ~20% of the excess arrival delay"); fig.savefig(out / "2c_dot_weather_vs_excess.png"); plt.close(fig)
print("\nDone.")
