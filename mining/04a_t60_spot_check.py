"""Step 4 pre-check: print per-leg timelines (local clock, absolute minutes, T-60 status) for 3 aircraft-days and
re-derive the absolute times independently in pandas, asserting they match propagation_t60 (built in SQL)."""
import numpy as np, pandas as pd
from common import connect, ensure_propagation_t60, outdir

ensure_propagation_t60()
con = connect(); out = outdir("model_comparison")
CASES = [("N841VA", 20150202, "East->West coast chain (JFK -> SFO -> ...): arrival clock is 3 h behind departure clock"),
         ("N523JB", 20150518, "Predecessor lands after midnight while the next flight is scheduled before midnight"),
         ("N470AA", 20150909, "A full day with all four T-60 statuses")]
hhmm = lambda m: f"{int(m // 60) % 24:02d}:{int(m % 60):02d}"
def absfmt(a):   # absolute minute -> 'MM-DD hh:mm' (axis: days since 2015-01-01 * 1440 + minute)
    d = pd.Timestamp("2015-01-01") + pd.Timedelta(minutes=float(a)); return d.strftime("%m-%d %H:%M")
lines = []
P = lambda s="": (print(s), lines.append(s))
for tail, dk, why in CASES:
    df = con.execute(f"""SELECT t.leg_number_of_day AS leg, ao.airport_iata_code AS orig, ad.airport_iata_code AS dest, f.scheduled_departure_min AS sdep,
        f.departure_delay_min AS dd, f.scheduled_arrival_min AS sarr, f.arrival_min AS arr_clock, f.elapsed_time_min AS el, f.scheduled_time_min AS sched_el,
        f.cancelled_flag AS canc, f.diverted_flag AS div, f.arrival_delay_min AS ad, t.prev_leg_status, t.prev_leg_arr_delay, t.prev_leg_dep_delay,
        t.prev_leg_sched_arr_minus_t60 AS sched_arr_m_t60, t.planned_turn_min AS planned, t.t60_abs, t.sdep_abs, t.prev_arr_abs, t.prev_out_abs
        FROM propagation_t60 t JOIN fact_flight f USING(flight_key) JOIN dim_airport ao ON ao.airport_key=f.origin_airport_key
        JOIN dim_airport ad ON ad.airport_key=f.destination_airport_key
        WHERE t.tail_number='{tail}' AND t.date_key={dk} ORDER BY t.leg_number_of_day""").df()
    di = (pd.Timestamp(str(dk)) - pd.Timestamp("2015-01-01")).days
    flown = (df.canc == 0) & (df["div"] == 0)
    df["sdep_abs_chk"] = di * 1440 + df.sdep
    k = np.round((df.sdep + df.dd + df.el - df.arr_clock) / 1440.0)
    df["arr_abs_chk"] = np.where(flown, di * 1440 + df.arr_clock + 1440 * k, np.nan)           # gate-in, clock of destination
    df["tz_shift_h"] = (df.arr_abs_chk - (df.sdep_abs_chk + df.dd + df.el)) / 60               # dest clock minus origin clock
    df["out_dest_clock"] = df.arr_abs_chk - df.el                                              # gate-out expressed in destination clock
    P(f"\n{'=' * 118}\n{tail}  {dk}   {why}\n{'=' * 118}")
    P("LOCAL CLOCKS (each airport's own time) and ABSOLUTE axis (clock of the airport where the aircraft is on the ground)")
    for _, r in df.iterrows():
        P(f"leg {int(r.leg)}: {r.orig}->{r.dest}  sched dep {hhmm(r.sdep)} (dep delay {r.dd:+.0f}) actual dep {hhmm(r.sdep + r.dd)} | sched arr {hhmm(r.sarr)} "
          f"actual arr {hhmm(r.arr_clock)} (arr delay {r.ad:+.0f}) | gate-to-gate {r.el:.0f} min, tz shift {r.tz_shift_h:+.1f} h"
          if flown[_] else f"leg {int(r.leg)}: {r.orig}->{r.dest}  sched dep {hhmm(r.sdep)}  CANCELLED/DIVERTED")
        if flown[_]: P(f"         abs: sched dep {absfmt(r.sdep_abs_chk)} | gate-in {absfmt(r.arr_abs_chk)} (dest clock) | gate-out in dest clock {absfmt(r.out_dest_clock)}")
    P("\nT-60 STATUS OF THE PREDECESSOR (as seen 60 min before each leg's scheduled departure)")
    for i, r in df.iterrows():
        if i == 0: P(f"leg {int(r.leg)}: status={r.prev_leg_status}  (no predecessor)"); continue
        p = df.iloc[i - 1]
        P(f"leg {int(r.leg)}: T-60 = {absfmt(r.t60_abs)} | predecessor {p.orig}->{p.dest}: gate-out {absfmt(p.out_dest_clock) if flown[i-1] else 'n/a'}, gate-in {absfmt(p.arr_abs_chk) if flown[i-1] else 'n/a'}"
          f" => status={r.prev_leg_status:<12} prev_arr_delay={r.prev_leg_arr_delay if pd.notna(r.prev_leg_arr_delay) else 'NaN'} prev_dep_delay={r.prev_leg_dep_delay if pd.notna(r.prev_leg_dep_delay) else 'NaN'}"
          f" sched_arr-T60={r.sched_arr_m_t60:+.0f} planned_turn={r.planned:.0f}")
        # independent re-derivation of status and assertions against the SQL table
        if flown[i - 1] and p.dest == r.orig:
            st = "landed" if p.arr_abs_chk <= r.t60_abs else ("airborne" if p.out_dest_clock <= r.t60_abs else "not_departed")
            assert st == r.prev_leg_status, (tail, dk, i, st, r.prev_leg_status)
            assert abs(p.arr_abs_chk - r.prev_arr_abs) < 1e-6 and abs(p.out_dest_clock - r.prev_out_abs) < 1e-6
            assert abs(r.t60_abs - (r.sdep_abs - 60)) < 1e-6 and abs(r.sdep_abs - r.sdep_abs_chk) < 1e-6
    P("\nAssertions passed: independent pandas re-derivation of absolute times and T-60 status equals the SQL table for every leg.")
(out / "t60_spot_check.txt").write_text("\n".join(lines))
