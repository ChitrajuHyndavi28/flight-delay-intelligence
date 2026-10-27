"""Step 1 - OLAP results export (O2). Tables + charts -> results/olap/.

Definitions (stated on every slide-ready table):
  flown        = not cancelled and not diverted (delay metrics use flown flights only)
  delayed      = arrival_delay_min >= 15  (DOT rule; NOTE warehouse arrival_delayed_15_flag uses > 15 and is NOT used);
               on-time % = 100 - delayed %
  cancel rate  = cancelled / all scheduled flights
  Time-only cuts (month, weekday, hour, airline) use all 12 months; airport-level cuts exclude
  October (airport codes unmapped in the ETL -> airport_key = 0).
  Hour buckets (scheduled departure): morning 05-11, afternoon 12-17, evening 18-21, night 22-04.
"""
import numpy as np, pandas as pd
from scipy import stats
from common import connect, outdir, set_plot_style, SEED

con = connect(); out = outdir("olap"); plt = set_plot_style()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
BUCKET = """CASE WHEN f.scheduled_departure_min//60 BETWEEN 5 AND 11 THEN '1 Morning (05-11)'
 WHEN f.scheduled_departure_min//60 BETWEEN 12 AND 17 THEN '2 Afternoon (12-17)'
 WHEN f.scheduled_departure_min//60 BETWEEN 18 AND 21 THEN '3 Evening (18-21)'
 ELSE '4 Night (22-04)' END"""
METRICS = """count(*) AS flights,
  round(100.0*sum(CASE WHEN cancelled_flag=0 AND diverted_flag=0 AND arrival_delay_min<15 THEN 1 ELSE 0 END)
        / nullif(sum(CASE WHEN cancelled_flag=0 AND diverted_flag=0 THEN 1 END),0), 2) AS on_time_pct,
  round(avg(CASE WHEN cancelled_flag=0 AND diverted_flag=0 THEN arrival_delay_min END), 2) AS avg_arr_delay_min,
  round(avg(CASE WHEN cancelled_flag=0 AND diverted_flag=0 THEN departure_delay_min END), 2) AS avg_dep_delay_min,
  round(100.0*avg(cancelled_flag), 2) AS cancel_rate_pct"""
FROM = "fact_flight f JOIN dim_date d USING(date_key)"

def q(sql): return con.execute(sql).df()
def save(df, name): df.to_csv(out / f"{name}.csv", index=False); print(f"\n--- {name} ---"); print(df.to_string(index=False))

def three_panel(df, x, title, fname, xlabels=None, rot=0):
    """4 panels. Bars (zero baseline) for delays/cancellations; on-time % as a dot/line plot (no truncated bars)."""
    xs = df[x].astype(str) if xlabels is None else xlabels
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8))
    ax = axs[0]
    if x == "month": ax.plot(xs, df.on_time_pct, "o-", color="#2a7ab0", lw=2, ms=7)
    else: ax.plot(xs, df.on_time_pct, "o", color="#2a7ab0", ms=11)
    for xi, v in zip(xs, df.on_time_pct): ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.set_ylabel("On-time arrivals (%, delay < 15 min)"); ax.set_title("On-time arrivals (%)")
    ax.set_ylim(df.on_time_pct.min() - 4, df.on_time_pct.max() + 4)
    for ax, col, lab, c in [(axs[1], "avg_arr_delay_min", "Avg ARRIVAL delay (min)", "#d9822b"),
                            (axs[2], "avg_dep_delay_min", "Avg DEPARTURE delay (min)", "#8e6bbf"),
                            (axs[3], "cancel_rate_pct", "Cancellation rate (%)", "#b03a2e")]:
        ax.bar(xs, df[col], color=c); ax.set_ylabel(lab); ax.set_title(lab); ax.axhline(0, color="k", lw=.8)
        ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=9, padding=2)
    for ax in axs: ax.set_xlabel(title.split(" by ")[-1]); ax.tick_params(axis="x", rotation=rot)
    fig.suptitle(title, fontsize=15); fig.savefig(out / fname); plt.close(fig)

# ---- time cuts ----------------------------------------------------------------------------
mon = q(f"SELECT d.month_number AS month, {METRICS} FROM {FROM} GROUP BY 1 ORDER BY 1"); save(mon, "by_month")
three_panel(mon, "month", "OLAP: delay performance by month (2015)", "by_month.png")

dow = q(f"SELECT d.day_of_week_number AS dow_num, d.day_of_week_name AS day_of_week, {METRICS} FROM {FROM} GROUP BY 1,2 ORDER BY 1")
save(dow, "by_day_of_week")
three_panel(dow, "day_of_week", "OLAP: delay performance by day of week", "by_day_of_week.png", xlabels=dow.day_of_week.str[:3])

bk = q(f"SELECT {BUCKET} AS dep_bucket, {METRICS} FROM {FROM} GROUP BY 1 ORDER BY 1"); bk["dep_bucket"] = bk.dep_bucket.str[2:]
save(bk, "by_dep_bucket")
m_ = bk.iloc[0]; print(f"\nMORNING bucket: avg ARRIVAL delay = {m_.avg_arr_delay_min:.2f} min; avg DEPARTURE delay = {m_.avg_dep_delay_min:.2f} min (arrival < departure: schedule padding/make-up in air)")
three_panel(bk, "dep_bucket", "OLAP: delay performance by scheduled departure bucket", "by_dep_bucket.png", rot=15)

hr = q(f"SELECT f.scheduled_departure_min//60 AS dep_hour, {METRICS} FROM {FROM} GROUP BY 1 ORDER BY 1"); save(hr, "by_dep_hour")
fig, ax = plt.subplots(figsize=(11, 4.8)); ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax.plot(hr.dep_hour, hr.avg_arr_delay_min, "o-", color="#d9822b", label="Avg arrival delay (min)")
ax2.plot(hr.dep_hour, hr.on_time_pct, "s--", color="#2a7ab0", label="On-time %")
ax.set_xlabel("Scheduled departure hour"); ax.set_ylabel("Avg arrival delay (min)"); ax2.set_ylabel("On-time arrivals (%)")
ax.set_xticks(range(0, 24)); ax.set_title("OLAP: delay builds through the day (by scheduled departure hour)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left")
fig.savefig(out / "by_dep_hour.png"); plt.close(fig)

# ---- airports (October excluded) ----------------------------------------------------------
ap = q("""SELECT a.airport_iata_code AS airport, a.city, a.state, count(*) AS flights,
  round(avg(departure_delay_min),2) AS avg_dep_delay_min, round(avg(arrival_delay_min),2) AS avg_arr_delay_min,
  round(100.0*avg((arrival_delay_min>=15)::int),2) AS delayed_pct
  FROM fact_flight f JOIN dim_date d USING(date_key) JOIN dim_airport a ON a.airport_key=f.origin_airport_key
  WHERE f.cancelled_flag=0 AND f.diverted_flag=0 AND f.origin_airport_key<>0 AND d.month_number<>10
  GROUP BY 1,2,3 HAVING count(*)>=5000""")
ap = ap.sort_values("avg_dep_delay_min", ascending=False)
print(f"\n{len(ap)} airports with >=5,000 flown departures (Oct excluded)")
save(ap.head(15), "airports_top15_worst_dep_delay"); save(ap.tail(15).iloc[::-1], "airports_bottom15_best_dep_delay")
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, d, t, c in [(axs[0], ap.head(15).iloc[::-1], "15 worst airports", "#b03a2e"), (axs[1], ap.tail(15), "15 best airports", "#2a7ab0")]:
    ax.barh(d.airport, d.avg_dep_delay_min, color=c); ax.set_xlabel("Avg departure delay (min)"); ax.set_ylabel("Origin airport")
    ax.set_title(t)
fig.suptitle("Airports by avg departure delay (>=5,000 flights, Oct excluded)", fontsize=15); fig.savefig(out / "airports_top_bottom15.png"); plt.close(fig)

# ---- airlines -----------------------------------------------------------------------------
al = q(f"SELECT l.airline_iata_code AS airline, l.airline_name, {METRICS} FROM {FROM} JOIN dim_airline l USING(airline_key) GROUP BY 1,2")
al = al.sort_values("on_time_pct", ascending=False); save(al, "airline_ranking")
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
d = al.iloc[::-1]
axs[0].plot(d.on_time_pct, d.airline, "o", color="#2a7ab0", ms=9); axs[0].grid(axis="x", alpha=.3)
for v, a_ in zip(d.on_time_pct, d.airline): axs[0].annotate(f"{v:.1f}", (v, a_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
axs[0].set_xlabel("On-time arrivals (%, delay < 15 min)"); axs[0].set_ylabel("Airline")
axs[0].set_title("On-time %")
d = al.sort_values("cancel_rate_pct")
axs[1].barh(d.airline, d.cancel_rate_pct, color="#b03a2e"); axs[1].set_xlabel("Cancellation rate (%)"); axs[1].set_ylabel("Airline")
axs[1].set_title("Cancellation rate (sorted best to worst, top = best)"); axs[1].invert_yaxis()
fig.suptitle("Airline ranking (2015, all months)", fontsize=15); fig.savefig(out / "airline_ranking.png"); plt.close(fig)

pv = q(f"""SELECT l.airline_iata_code AS airline, d.month_number AS m, avg(arrival_delay_min) AS v FROM {FROM}
  JOIN dim_airline l USING(airline_key) WHERE cancelled_flag=0 AND diverted_flag=0 GROUP BY 1,2""").pivot(index="airline", columns="m", values="v")
pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]; pv.round(2).reset_index().to_csv(out / "airline_x_month_avg_arr_delay.csv", index=False)
print("\n--- airline_x_month_avg_arr_delay (min) ---"); print(pv.round(1).to_string())
import seaborn as sns
fig, ax = plt.subplots(figsize=(12, 6.5))
sns.heatmap(pv, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg arrival delay (min)"})
ax.set_xlabel("Month (2015)"); ax.set_ylabel("Airline"); ax.set_title("Avg arrival delay (min): airline x month")
fig.savefig(out / "airline_x_month_heatmap.png"); plt.close(fig)

# ---- evening vs morning: Mann-Whitney U + effect sizes -------------------------------------
print("\n=== Evening (18-21) vs Morning (05-11) departure delay: Mann-Whitney U ===")
x = q(f"""SELECT {BUCKET} AS b, f.departure_delay_min AS dd FROM {FROM} WHERE cancelled_flag=0 AND diverted_flag=0
   AND f.scheduled_departure_min//60 BETWEEN 5 AND 21 AND NOT f.scheduled_departure_min//60 BETWEEN 12 AND 17""")
mo = x.loc[x.b.str.startswith("1"), "dd"].to_numpy(np.float32); ev = x.loc[x.b.str.startswith("3"), "dd"].to_numpy(np.float32)
U, p = stats.mannwhitneyu(ev, mo, alternative="two-sided", method="asymptotic")
n1, n2 = len(ev), len(mo)
cles = U / (n1 * n2)                      # P(random evening flight is delayed more than random morning flight) + 0.5*ties
rbc = 2 * cles - 1                        # rank-biserial correlation (= Cliff's delta)
rng = np.random.default_rng(SEED)         # full-size bootstrap (200 resamples) of exact delta
bs = []
for _ in range(200):
    a = rng.choice(ev, n1); b = rng.choice(mo, n2)
    bs.append(2 * stats.mannwhitneyu(a, b, method="asymptotic").statistic / (n1 * n2) - 1)
mw = pd.DataFrame([{
    "n_evening": n1, "n_morning": n2, "median_evening": np.median(ev), "median_morning": np.median(mo),
    "mean_evening": ev.mean(), "mean_morning": mo.mean(), "mean_diff_min": ev.mean() - mo.mean(),
    "pct_dep_delayed15_evening": 100 * (ev >= 15).mean(), "pct_dep_delayed15_morning": 100 * (mo >= 15).mean(),
    "U": U, "p_value": p, "CLES_prob_evening_worse": cles, "cliffs_delta_exact_2U/(n1n2)-1": rbc,
    "delta_boot95_lo_fullsize_200": np.percentile(bs, 2.5), "delta_boot95_hi_fullsize_200": np.percentile(bs, 97.5),
    "cohens_d_raw": (ev.mean() - mo.mean()) / np.sqrt((ev.var() + mo.var()) / 2)}]).T.rename(columns={0: "value"})
save(mw.reset_index().rename(columns={"index": "metric"}), "evening_vs_morning_mannwhitney")
mag = "negligible" if abs(rbc) < .147 else "small" if abs(rbc) < .33 else "medium" if abs(rbc) < .474 else "large"
print(f"\nInterpretation: p={p:.2g} (uninformative at n={n1+n2:,}); Cliff's delta={rbc:.3f} -> {mag} effect "
      f"(Romano thresholds .147/.33/.474); P(evening > morning)={cles:.3f}")
fig, ax = plt.subplots(figsize=(9, 4.8))
bins = np.arange(-30, 121, 5)
ax.hist(np.clip(mo, -30, 120), bins, density=True, alpha=.6, label=f"Morning (median {np.median(mo):.0f} min)", color="#2a7ab0")
ax.hist(np.clip(ev, -30, 120), bins, density=True, alpha=.6, label=f"Evening (median {np.median(ev):.0f} min)", color="#d9822b")
ax.set_xlabel("Departure delay (min, clipped to [-30, 120])"); ax.set_ylabel("Density")
ax.set_title(f"Evening vs morning departure delay: Cliff's delta = {rbc:.2f} ({mag})"); ax.legend()
fig.savefig(out / "evening_vs_morning_hist.png"); plt.close(fig)
