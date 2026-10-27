"""Step 3 - Delay propagation via aircraft tail_number (O4). Outputs -> results/propagation/.

Builds table propagation_features in flight_dw.duckdb with LAG window functions (see common.PROPAGATION_SQL):
  prev_ok  = analysis set (previous leg same aircraft & same day, flown, destination == this origin, known airports)
  feat_ok  = prev_ok AND previous actual arrival <= this scheduled departure -> candidate model features
             prev_leg_arr_delay, turnaround_min (known at prediction time; NULL otherwise)
October is excluded automatically (airport_key = 0 -> cannot verify continuity).
Analyses use the analysis set (turnaround may be negative); the saved features use the feature set only.
"""
import numpy as np, pandas as pd
from common import connect, ensure_propagation_table, outdir, set_plot_style, SEED

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
import sys
ensure_propagation_table(force="--rebuild" in sys.argv)      # normally just ensures the table exists (auto-rebuild if missing)
con = connect(); out = outdir("propagation"); plt = set_plot_style()
q = lambda s: con.execute(s).df()
def save(df, name, show=True):
    df.to_csv(out / f"{name}.csv", index=False)
    if show: print(f"\n--- {name} ---"); print(df.to_string(index=False))

def case(col, edges, labels):
    parts = [f"WHEN {col} < {e} THEN '{i:02d} {l}'" for i, (e, l) in enumerate(zip(edges, labels))]
    return f"CASE {' '.join(parts)} ELSE '{len(edges):02d} {labels[-1]}' END"
IN_EDGES = [-15, 0, 15, 30, 60, 90, 120, 180]
IN_LAB = ["< -15 (early)", "-15..-1", "0..14 (on time)", "15..29", "30..59", "60..89", "90..119", "120..179", "180+"]
IN_BIN = case("p.prev_arr_delay_any", IN_EDGES, IN_LAB)
BUF_BIN = case("p.planned_turn_min", [30, 45, 60, 90, 120, 180], ["<30", "30-44", "45-59", "60-89", "90-119", "120-179", "180+"][:6] + ["180+"])
HR = "CASE WHEN f.scheduled_departure_min//60 BETWEEN 5 AND 11 THEN 'morning' WHEN f.scheduled_departure_min//60 BETWEEN 12 AND 17 THEN 'afternoon' ELSE 'evening/night' END"
BASE = "propagation_features p JOIN fact_flight f USING(flight_key)"
NOTE = "airport-known months only (Oct excluded)"

# ---- 0. funnel ---------------------------------------------------------------------------------
print("=== Step 3 funnel ===")
fn = q("""SELECT count(*) AS flights_with_tail, sum(has_prev::int) AS has_prev_leg_same_day, sum(prev_flown_flag::int) AS prev_leg_flown,
  sum(continuity_ok::int) AS continuity_ok_dest_eq_origin, sum(prev_ok::int) AS analysis_set, sum(feat_ok::int) AS feature_set FROM propagation_features""")
fn = fn.T.reset_index(); fn.columns = ["stage", "flights"]; fn["pct_of_flights_with_tail"] = (100 * fn.flights / fn.flights.iloc[0]).round(1); save(fn, "3_funnel")
print(q("""SELECT sum((continuity_ok AND NOT prev_ok AND origin_airport_key<>0)::int) AS continuity_ok_but_dropped_known_airports,
  sum((prev_ok AND NOT feat_ok)::int) AS prev_arrived_after_sched_dep_not_in_feature_set FROM propagation_features""").to_string(index=False))
print("Tail-days with ties in scheduled departure are broken by flight_number, flight_key (~1,300 rows).")

# ---- A. departure delay vs previous-leg arrival delay -------------------------------------------
print("\n=== A. Departure delay vs previous-leg arrival delay ===")
A = q(f"""SELECT {IN_BIN} AS in_bin, count(*) AS flights, round(avg(p.prev_arr_delay_any),1) AS mean_incoming_delay,
  round(avg(f.departure_delay_min),2) AS mean_dep_delay, round(median(f.departure_delay_min),1) AS median_dep_delay,
  round(100*avg((f.departure_delay_min>=15)::int),2) AS pct_dep_delayed15, round(stddev(f.departure_delay_min)/sqrt(count(*)),3) AS se_mean
  FROM {BASE} WHERE p.prev_ok AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1 ORDER BY 1""")
A["incoming_delay_bin"] = A.in_bin.str[3:]
base1 = q(f"""SELECT count(*) AS flights, round(avg(f.departure_delay_min),2) AS mean_dep_delay, round(100*avg((f.departure_delay_min>=15)::int),2) AS pct_dep_delayed15
  FROM {BASE} WHERE p.leg_number=1 AND p.origin_airport_key<>0 AND f.cancelled_flag=0 AND f.diverted_flag=0""").iloc[0]
save(A.drop(columns="in_bin"), "3A_dep_delay_vs_incoming_delay")
print(f"Baseline: first leg of the day (no incoming aircraft): n={int(base1.flights):,}, mean dep delay {base1.mean_dep_delay} min, {base1.pct_dep_delayed15}% delayed >= 15")
Ah = q(f"""SELECT {IN_BIN} AS in_bin, {HR} AS dep_period, count(*) AS flights, round(avg(f.departure_delay_min),2) AS mean_dep_delay
  FROM {BASE} WHERE p.prev_ok AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1,2 ORDER BY 1,2"""); Ah["incoming_delay_bin"] = Ah.in_bin.str[3:]
save(Ah.drop(columns="in_bin"), "3A_dep_delay_vs_incoming_delay_by_period", show=False)
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
ax = axs[0]; xi = np.arange(len(A))
ax.errorbar(xi, A.mean_dep_delay, yerr=1.96 * A.se_mean, fmt="o-", color="#2a7ab0", capsize=3, lw=2, label="Mean departure delay (95% CI)")
ax.axhline(base1.mean_dep_delay, color="#888", ls="--", label=f"First leg of day, no incoming aircraft ({base1.mean_dep_delay:.1f} min)")
ax.set_xticks(xi); ax.set_xticklabels(A.incoming_delay_bin, rotation=30, ha="right"); ax.set_xlabel("Previous leg arrival delay (min)"); ax.set_ylabel("Mean departure delay of this flight (min)")
ax.set_title("A late inbound aircraft becomes a late departure"); ax.legend(frameon=False, fontsize=9); ax.set_ylim(bottom=min(0, A.mean_dep_delay.min() - 2))
ax = axs[1]
for per, c in [("morning", "#2a7ab0"), ("afternoon", "#d9822b"), ("evening/night", "#8e6bbf")]:
    d = Ah[Ah.dep_period == per]; ax.plot([A.incoming_delay_bin.tolist().index(b) for b in d.incoming_delay_bin], d.mean_dep_delay, "o-", color=c, label=per)
ax.set_xticks(xi); ax.set_xticklabels(A.incoming_delay_bin, rotation=30, ha="right"); ax.set_xlabel("Previous leg arrival delay (min)"); ax.set_ylabel("Mean departure delay (min)")
ax.set_title("Same curve by scheduled departure period (hour mix held fixed)"); ax.legend(frameon=False, title="This flight departs")
fig.suptitle(f"Delay propagation curve ({NOTE})", fontsize=15); fig.savefig(out / "3A_propagation_curve.png"); plt.close(fig)

share = q(f"""SELECT count(*) AS dep_delayed15_flights_leg2plus_in_analysis_set,
  sum((p.prev_arr_delay_any>=15)::int) AS after_late_inbound, round(100*avg((p.prev_arr_delay_any>=15)::int),1) AS pct_after_late_inbound,
  round(100*sum((p.prev_arr_delay_any>=15)::int)/sum(1),1) AS dummy FROM {BASE} WHERE p.prev_ok AND f.departure_delay_min>=15 AND f.cancelled_flag=0 AND f.diverted_flag=0""").drop(columns="dummy")
save(share, "3A_share_of_departure_delays_after_late_inbound")

# ---- B. turnaround buffer ---------------------------------------------------------------------
print("\n=== B. Turnaround buffer: how much of an incoming delay does a long turn absorb? ===")
IN2 = case("p.prev_arr_delay_any", [15, 30, 60, 120], ["< 15 (none)", "15-29", "30-59", "60-119", "120+"])
B = q(f"""SELECT {BUF_BIN} AS buf_bin, {IN2} AS in_bin, count(*) AS flights, round(avg(p.prev_arr_delay_any),1) AS mean_incoming_delay,
  round(avg(f.departure_delay_min),2) AS mean_dep_delay, round(100*avg((f.departure_delay_min>=15)::int),1) AS pct_dep_delayed15
  FROM {BASE} WHERE p.prev_ok AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1,2 ORDER BY 1,2""")
B["planned_buffer_min"] = B.buf_bin.str[3:]; B["incoming_delay"] = B.in_bin.str[3:]
B["absorbed_min"] = (B.mean_incoming_delay - B.mean_dep_delay).where(~B.in_bin.str.startswith("00")).round(2)
B["pct_absorbed"] = (100 * (1 - B.mean_dep_delay / B.mean_incoming_delay)).where(~B.in_bin.str.startswith("00")).round(1)
Bs = B.drop(columns=["buf_bin", "in_bin"]); save(Bs[["planned_buffer_min", "incoming_delay", "flights", "mean_incoming_delay", "mean_dep_delay", "pct_dep_delayed15", "absorbed_min", "pct_absorbed"]], "3B_buffer_absorption")
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2)); bl = sorted(B.buf_bin.unique()); bn = [b[3:] for b in bl]
for inb, c in zip(sorted(B.in_bin.unique()), ["#999999", "#2a7ab0", "#d9822b", "#b03a2e", "#6b1f1f"]):
    d = B[B.in_bin == inb].set_index("buf_bin").reindex(bl)
    axs[0].plot(range(len(bl)), d.mean_dep_delay, "o-", color=c, label=f"incoming {inb[3:]} min")
    if not inb.startswith("00"): axs[1].plot(range(len(bl)), d.pct_absorbed, "o-", color=c, label=f"incoming {inb[3:]} min")
for ax, yl in zip(axs, ["Mean departure delay of this flight (min)", "Share of incoming delay absorbed (%)"]):
    ax.set_xticks(range(len(bl))); ax.set_xticklabels(bn); ax.set_xlabel("Planned turnaround (this sched. departure - previous sched. arrival), min"); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=9)
axs[0].set_title("Departure delay by buffer and incoming delay"); axs[0].set_ylim(bottom=0); axs[1].set_title("Longer scheduled turns absorb more"); axs[1].axhline(0, color="k", lw=.8)
fig.suptitle("Turnaround buffer and delay absorption", fontsize=15); fig.savefig(out / "3B_buffer_absorption.png"); plt.close(fig)

# ---- C. amplification by leg number --------------------------------------------------------------
print("\n=== C. Delay through the aircraft's day ===")
LEGC = "least(p.leg_number, 8)"
C = q(f"""SELECT {LEGC} AS leg, count(*) AS flights, round(avg(f.departure_delay_min),2) AS mean_dep_delay, round(avg(f.arrival_delay_min),2) AS mean_arr_delay,
  round(100*avg((f.arrival_delay_min>=15)::int),2) AS pct_arr_delayed15,
  round(avg(CASE WHEN p.prev_ok AND p.prev_arr_delay_any>=15 THEN f.departure_delay_min - p.prev_arr_delay_any END),2) AS mean_ground_change_late_inbound,
  round(avg(f.arrival_delay_min - f.departure_delay_min),2) AS mean_air_change
  FROM {BASE} WHERE p.origin_airport_key<>0 AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1 ORDER BY 1"""); C["leg"] = C.leg.astype(int)
C["leg_label"] = C.leg.astype(str).where(C.leg < 8, "8+")
save(C, "3C_by_leg_number")
Ch = q(f"""SELECT {LEGC} AS leg, {HR} AS dep_period, count(*) AS flights, round(avg(f.departure_delay_min),2) AS mean_dep_delay
  FROM {BASE} WHERE p.origin_airport_key<>0 AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1,2 ORDER BY 1,2""")
save(Ch, "3C_by_leg_and_period", show=False)
# persistence: aircraft-days whose first leg arrived badly late vs on time
P = q(f"""WITH l1 AS (SELECT tail_number, date_key, CASE WHEN arrival_delay_min>=60 THEN 'leg 1 arrived 60+ min late' WHEN arrival_delay_min>=15 THEN 'leg 1 arrived 15-59 late'
    ELSE 'leg 1 on time' END AS leg1_status FROM propagation_features WHERE leg_number=1 AND flown AND origin_airport_key<>0)
  SELECT l1.leg1_status, {LEGC} AS leg, count(*) AS flights, round(avg(f.arrival_delay_min),2) AS mean_arr_delay
  FROM {BASE} JOIN l1 ON l1.tail_number=p.tail_number AND l1.date_key=p.date_key
  WHERE p.leg_number BETWEEN 1 AND 6 AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1,2 ORDER BY 1,2""")
save(P, "3C_persistence_after_leg1")
fig, axs = plt.subplots(1, 3, figsize=(19, 5))
axs[0].plot(C.leg_label, C.mean_dep_delay, "o-", color="#8e6bbf", lw=2, label="Departure delay"); axs[0].plot(C.leg_label, C.mean_arr_delay, "s-", color="#d9822b", lw=2, label="Arrival delay")
axs[0].set_xlabel("Leg of the aircraft's day (scheduled order)"); axs[0].set_ylabel("Mean delay (min)"); axs[0].set_title("Delay grows through the day"); axs[0].legend(frameon=False); axs[0].set_ylim(bottom=0)
w = .38; xi = np.arange(len(C))
axs[1].bar(xi - w / 2, C.mean_ground_change_late_inbound.fillna(0), w, color="#2a7ab0", label="On the ground, late-inbound turns only (dep delay - previous arr delay)"); axs[1].bar(xi + w / 2, C.mean_air_change, w, color="#6aa84f", label="In the air (arr delay - dep delay)")
axs[1].axhline(0, color="k", lw=.8); axs[1].set_xticks(xi); axs[1].set_xticklabels(C.leg_label); axs[1].set_xlabel("Leg of the day"); axs[1].set_ylabel("Mean change in delay (min)")
axs[1].set_title("Where delay is gained (+) or recovered (-)\n(ground bars limited to inbound >= 15 min late: early arrivals cannot leave early)", fontsize=11); axs[1].legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.0))
axs[1].set_ylim(-22, 2)
for st, c in zip(sorted(P.leg1_status.unique()), ["#b03a2e", "#d9822b", "#2a7ab0"]):
    d = P[P.leg1_status == st]; axs[2].plot(d.leg, d.mean_arr_delay, "o-", color=c, lw=2, label=st)
axs[2].set_xlabel("Leg of the day"); axs[2].set_ylabel("Mean arrival delay (min)"); axs[2].set_title("A bad first leg echoes through the day"); axs[2].legend(frameon=False, fontsize=9); axs[2].axhline(0, color="k", lw=.8)
fig.suptitle("Delay amplification along the aircraft's day (Oct excluded)", fontsize=15); fig.savefig(out / "3C_amplification.png"); plt.close(fig)

# ---- D. absorb vs amplify: airlines and airports (observed minus expected) -------------------------
print("\n=== D. Who absorbs vs amplifies incoming delay? ===")
D0 = q(f"""SELECT f.flight_key, l.airline_iata_code AS airline, a.airport_iata_code AS airport, f.departure_delay_min AS dd, p.prev_arr_delay_any AS inc,
  {case('p.prev_arr_delay_any', [30, 60, 120], ['15-29', '30-59', '60-119', '120+'])} AS inb, {BUF_BIN} AS bb, {HR} AS per
  FROM {BASE} JOIN dim_airline l ON l.airline_key=f.airline_key JOIN dim_airport a ON a.airport_key=f.origin_airport_key
  WHERE p.prev_ok AND p.prev_arr_delay_any>=15 AND f.cancelled_flag=0 AND f.diverted_flag=0""")
D0["expected_dd"] = D0.groupby(["inb", "bb", "per"]).dd.transform("mean")
D0["oe"] = D0.dd - D0.expected_dd; D0["absorbed"] = D0.inc - D0.dd
print(f"{len(D0):,} turns with incoming delay >= 15 min (analysis set)")
def summarize(g, key, min_n):
    r = g.groupby(key).agg(turns=("dd", "size"), mean_incoming_delay=("inc", "mean"), mean_dep_delay=("dd", "mean"), absorbed_min=("absorbed", "mean"),
                           observed_minus_expected=("oe", "mean"), se=("oe", lambda x: x.std() / np.sqrt(len(x)))).reset_index()
    r["pct_absorbed"] = 100 * r.absorbed_min / r.mean_incoming_delay; r = r[r.turns >= min_n].sort_values("observed_minus_expected")
    return r.round(2)
Dal = summarize(D0, "airline", 500); save(Dal, "3D_airline_absorption")
Dap = summarize(D0, "airport", 500); save(pd.concat([Dap.head(10), Dap.tail(10)]), "3D_airport_absorption_best10_worst10"); Dap.to_csv(out / "3D_airport_absorption_all.csv", index=False)
print(f"airports with >= 500 delayed-inbound turns: {len(Dap)}")
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
d = Dal.iloc[::-1]; axs[0].barh(d.airline, d.observed_minus_expected, xerr=1.96 * d.se, color=np.where(d.observed_minus_expected < 0, "#2a7ab0", "#b03a2e"), capsize=3)
axs[0].axvline(0, color="k", lw=.8); axs[0].set_xlabel("Departure delay vs expected for same incoming delay, buffer & time of day (min)"); axs[0].set_ylabel("Airline")
axs[0].set_title("Airlines: absorbing (left) vs amplifying (right)"); axs[0].invert_yaxis()
d = pd.concat([Dap.head(10), Dap.tail(10)]).iloc[::-1]; axs[1].barh(d.airport, d.observed_minus_expected, xerr=1.96 * d.se, color=np.where(d.observed_minus_expected < 0, "#2a7ab0", "#b03a2e"), capsize=3)
axs[1].axvline(0, color="k", lw=.8); axs[1].set_xlabel("Observed - expected departure delay (min)"); axs[1].set_ylabel("Airport where the turn happens"); axs[1].invert_yaxis()
axs[1].set_title("Airports: 10 best and 10 worst at absorbing")
fig.suptitle("Who absorbs vs amplifies inbound delay (>=15 min late inbound; 95% CI)", fontsize=15); fig.savefig(out / "3D_absorb_amplify.png"); plt.close(fig)

# ---- E. validation vs DOT late_aircraft_delay_min ---------------------------------------------------
print("\n=== E. Validation vs DOT late_aircraft_delay_min ===")
E = q(f"""SELECT {IN2} AS in_bin, count(*) AS delayed_arrivals, round(100*avg((f.late_aircraft_delay_min>0)::int),1) AS pct_dot_late_aircraft,
  round(avg(f.late_aircraft_delay_min),1) AS mean_late_aircraft_min, round(avg(f.arrival_delay_min),1) AS mean_arr_delay,
  round(100*avg(f.late_aircraft_delay_min)/avg(f.arrival_delay_min),1) AS late_aircraft_share_of_delay_pct
  FROM {BASE} WHERE p.prev_ok AND f.arrival_delay_min>=15 AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1 ORDER BY 1""")
E["our_incoming_delay"] = E.in_bin.str[3:]; save(E.drop(columns="in_bin"), "3E_validation_vs_dot_by_incoming_bin")
L1 = q(f"""SELECT 'first leg of the day (no previous leg that day)' AS group_, count(*) AS delayed_arrivals, round(100*avg((f.late_aircraft_delay_min>0)::int),1) AS pct_dot_late_aircraft,
  round(avg(f.late_aircraft_delay_min),1) AS mean_late_aircraft_min FROM {BASE} WHERE p.leg_number=1 AND p.origin_airport_key<>0 AND f.arrival_delay_min>=15 AND f.cancelled_flag=0 AND f.diverted_flag=0""")
save(L1, "3E_first_leg_check")
cm = q(f"""SELECT (p.prev_arr_delay_any>=15) AS ours_prev_late, (f.late_aircraft_delay_min>0) AS dot_late_aircraft, count(*) AS n
  FROM {BASE} WHERE p.prev_ok AND f.arrival_delay_min>=15 AND f.cancelled_flag=0 AND f.diverted_flag=0 GROUP BY 1,2 ORDER BY 1,2""")
tp = cm[(cm.ours_prev_late) & (cm.dot_late_aircraft)].n.sum(); fp = cm[(cm.ours_prev_late) & (~cm.dot_late_aircraft)].n.sum()
fnn = cm[(~cm.ours_prev_late) & (cm.dot_late_aircraft)].n.sum(); tn = cm[(~cm.ours_prev_late) & (~cm.dot_late_aircraft)].n.sum()
agree = pd.DataFrame([dict(delayed_arrivals_in_analysis_set=int(cm.n.sum()), both_flag=int(tp), ours_only=int(fp), dot_only=int(fnn), neither=int(tn),
                           agreement_pct=round(100 * (tp + tn) / cm.n.sum(), 1), p_dot_given_ours=round(100 * tp / (tp + fp), 1), p_ours_given_dot=round(100 * tp / (tp + fnn), 1))])
save(agree, "3E_agreement_with_dot")
corr = q(f"""SELECT corr(p.prev_arr_delay_any, f.late_aircraft_delay_min) AS pearson_incoming_vs_late_aircraft,
  corr(p.prev_arr_delay_any, f.departure_delay_min) AS pearson_incoming_vs_dep_delay, count(*) AS n
  FROM {BASE} WHERE p.prev_ok AND f.arrival_delay_min>=15 AND f.cancelled_flag=0 AND f.diverted_flag=0""").round(3); save(corr, "3E_correlations")
tot = q(f"""SELECT sum(f.late_aircraft_delay_min) AS dot_late_aircraft_minutes_all,
  sum(CASE WHEN p.prev_ok AND p.prev_arr_delay_any>=15 THEN f.late_aircraft_delay_min END) AS in_flights_with_late_inbound_in_our_data
  FROM {BASE} WHERE f.arrival_delay_min>=15 AND f.cancelled_flag=0 AND f.diverted_flag=0 AND p.origin_airport_key<>0""")
tot["share_pct"] = round(100 * tot.in_flights_with_late_inbound_in_our_data / tot.dot_late_aircraft_minutes_all, 1); save(tot, "3E_late_aircraft_minutes_coverage")
fig, ax = plt.subplots(figsize=(9.5, 5)); xi = np.arange(len(E))
ax.bar(xi, E.mean_late_aircraft_min, color="#d9822b", label="Mean DOT late-aircraft minutes"); ax.bar(xi, E.mean_arr_delay - E.mean_late_aircraft_min, bottom=E.mean_late_aircraft_min, color="#cccccc", label="Other causes")
ax.set_xticks(xi); ax.set_xticklabels(E.our_incoming_delay); ax.set_xlabel("Our measured previous-leg arrival delay (min)"); ax.set_ylabel("Mean arrival delay of delayed flights (min)")
for x_, v in zip(xi, E.pct_dot_late_aircraft): ax.annotate(f"{v:.0f}% flagged\nlate-aircraft", (x_, E.mean_arr_delay.iloc[x_]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
ax.set_title("Validation: DOT late-aircraft delay tracks our measured incoming delay"); ax.legend(frameon=False, loc="upper left"); fig.savefig(out / "3E_validation_vs_dot.png"); plt.close(fig)

# ---- F. legacy 'known at scheduled departure' features (SUPERSEDED for modeling by the T-60 design in Step 4; kept for the record) --------------------------------------------------------------------
print("\n=== F. Candidate features for Step 4 (table propagation_features) ===")
F = q("""SELECT count(*) AS rows, count(legacy_prev_leg_arr_delay) AS legacy_prev_leg_arr_delay_nonnull, count(legacy_turnaround_min) AS legacy_turnaround_nonnull,
  round(100.0*count(legacy_prev_leg_arr_delay)/count(*),1) AS pct_with_feature, min(legacy_turnaround_min) AS min_turnaround, round(avg(legacy_prev_leg_arr_delay),2) AS mean_prev_arr_delay FROM propagation_features""")
save(F, "3F_feature_coverage")
c2 = con
c2.execute("COPY (SELECT flight_key, leg_number, legs_in_day, legacy_prev_leg_arr_delay, legacy_turnaround_min, planned_turn_min FROM propagation_features) TO '" + str(out / "propagation_features.parquet") + "' (FORMAT PARQUET)")
print("saved:", out / "propagation_features.parquet")
