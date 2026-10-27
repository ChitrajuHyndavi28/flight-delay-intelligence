"""Shared helpers: DB connection, paths, plotting defaults, seeds."""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
import os
DB_PATH = Path(os.environ.get("FLIGHT_DB", ROOT / "flight_dw.duckdb"))      # FLIGHT_DB overrides the path (used for safe tests on a scratch copy)
RESULTS = ROOT / "results"
SEED = 42


def connect(read_only=True):
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute("SET memory_limit='6GB'; SET threads=6; SET preserve_insertion_order=false;")
    return con


def outdir(name):
    p = RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
                         "axes.titlesize": 14, "axes.labelsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.autolayout": True})
    return plt


# Weather severity derivation (units in dim_weather: precipitation_mm, snowfall_mm, avg_wind_ms).
# Thresholds are the spec's (tenths-mm / tenths-m/s) converted to the warehouse's units:
#   severe   = SNOW > 25 mm OR PRCP > 25 mm (250 tenths) OR AWND > 13.4 m/s (134 tenths)
#   moderate = SNOW > 0    OR PRCP > 5 mm  (50 tenths)  OR AWND > 8.9 m/s  (89 tenths)
#   clear    = otherwise;  unknown = weather_key = 0 (no station within radius / unmapped airport)
# NULL measurements (e.g. station does not report SNOW) are treated as 0, never as evidence of weather.
def wx_class_sql(alias="w"):
    a = alias
    return f"""CASE WHEN {a}.weather_key = 0 THEN 'unknown'
      WHEN coalesce({a}.snowfall_mm,0) > 25 OR coalesce({a}.precipitation_mm,0) > 25
           OR coalesce({a}.avg_wind_ms,0) > 13.4 THEN 'severe'
      WHEN coalesce({a}.snowfall_mm,0) > 0 OR coalesce({a}.precipitation_mm,0) > 5
           OR coalesce({a}.avg_wind_ms,0) > 8.9 THEN 'moderate'
      ELSE 'clear' END"""


# ---------------------------------------------------------------------------------------------
# Delay-propagation table (Step 3). One row per flight with a non-null tail_number.
# Legs are ordered per (tail_number, scheduled date) by scheduled departure (ties: flight_number, flight_key),
# over ALL scheduled legs (cancelled/diverted included), so a cancelled previous leg breaks the chain
# instead of being skipped.  Times are local clocks at different airports, so everything is converted to an
# absolute minute axis (days since 2015-01-01 * 1440 + minute of day) in the clock of the airport where the
# aircraft is on the ground: actual arrival is aligned to the right day using actual departure + elapsed time
# (handles midnight rollover and time-zone shifts, |tz shift| < 12 h).
#   prev_ok      : previous leg same tail & same date, flown, prev destination = this origin, airports known,
#                  planned turn >= 0   -> ANALYSIS set (turnaround may be negative: aircraft arrived after this
#                  flight's scheduled departure)
#   feat_ok      : prev_ok AND previous actual arrival <= this scheduled departure -> value is known at
#                  prediction time -> prev_leg_arr_delay / turnaround_min are only filled here (FEATURE set)
# ---------------------------------------------------------------------------------------------
PROPAGATION_SQL = """
CREATE OR REPLACE TABLE propagation_features AS
WITH b AS (
  SELECT f.flight_key, f.tail_number, f.date_key, d.month_number AS month, f.airline_key,
         f.origin_airport_key AS o, f.destination_airport_key AS dst, f.scheduled_departure_min AS sdep,
         f.scheduled_arrival_min AS sarr, f.flight_number, (f.cancelled_flag=0 AND f.diverted_flag=0) AS flown,
         f.departure_delay_min AS dd, f.arrival_delay_min AS ad, f.arrival_min, f.elapsed_time_min, f.scheduled_time_min,
         date_diff('day', DATE '2015-01-01', d.full_date) AS di
  FROM fact_flight f JOIN dim_date d USING(date_key) WHERE f.tail_number IS NOT NULL),
c AS (
  SELECT *, di*1440 + sdep AS sdep_abs,
    CASE WHEN flown THEN di*1440 + arrival_min + 1440*round((sdep + dd + elapsed_time_min - arrival_min)/1440.0) END AS arr_abs,
    CASE WHEN scheduled_time_min IS NOT NULL THEN di*1440 + sarr + 1440*round((sdep + scheduled_time_min - sarr)/1440.0) END AS sarr_abs
  FROM b),
w AS (
  SELECT *, row_number() OVER p AS leg_number, count(*) OVER (PARTITION BY tail_number, date_key) AS legs_in_day,
    lag(flight_key) OVER p AS prev_key, lag(dst) OVER p AS prev_dst, lag(flown) OVER p AS prev_flown,
    lag(arr_abs) OVER p AS prev_arr_abs, lag(sarr_abs) OVER p AS prev_sarr_abs, lag(ad) OVER p AS prev_ad,
    lag(dd) OVER p AS prev_dd
  FROM c WINDOW p AS (PARTITION BY tail_number, date_key ORDER BY sdep, flight_number, flight_key)),
x AS (
  SELECT *, (prev_key IS NOT NULL AND prev_flown AND prev_dst = o AND o <> 0 AND prev_dst <> 0
             AND prev_sarr_abs IS NOT NULL AND sdep_abs - prev_sarr_abs >= 0) AS prev_ok
  FROM w)
SELECT flight_key, tail_number, date_key, month, airline_key, o AS origin_airport_key, flown, dd AS departure_delay_min,
  ad AS arrival_delay_min, leg_number, legs_in_day, (prev_key IS NOT NULL) AS has_prev, prev_key IS NOT NULL AND prev_flown AS prev_flown_flag,
  (prev_key IS NOT NULL AND prev_flown AND prev_dst = o) AS continuity_ok, prev_ok,
  CASE WHEN prev_ok THEN prev_ad END AS prev_arr_delay_any,
  CASE WHEN prev_ok THEN prev_dd END AS prev_dep_delay_any,
  CASE WHEN prev_ok THEN sdep_abs - prev_arr_abs END AS turnaround_actual_min,
  CASE WHEN prev_ok THEN sdep_abs - prev_sarr_abs END AS planned_turn_min,
  (prev_ok AND sdep_abs - prev_arr_abs >= 0) AS feat_ok,
  CASE WHEN prev_ok AND sdep_abs - prev_arr_abs >= 0 THEN prev_ad END AS legacy_prev_leg_arr_delay,
  CASE WHEN prev_ok AND sdep_abs - prev_arr_abs >= 0 THEN sdep_abs - prev_arr_abs END AS legacy_turnaround_min
FROM x
"""


def _ensure_table(table, sql, force=False):
    """Make sure a derived table exists. Existence is checked on a READ-ONLY connection; a write connection is opened only
    if the table is missing (or force=True), so normal runs never modify the database or fight a concurrent reader."""
    con = connect(read_only=True)
    exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [table]).fetchone()[0] > 0
    con.close()
    if exists and not force: return
    print(f"[common] derived table '{table}' {'is being rebuilt (force)' if exists else 'is missing -> rebuilding automatically'} ...", flush=True)
    try:
        con = connect(read_only=False)
    except duckdb.IOException as e:
        raise RuntimeError(f"Cannot rebuild '{table}': {DB_PATH.name} is locked by another process (close other Python/VS Code sessions "
                           f"using it, or restore backups/flight_dw_final.duckdb). Original error: {e}") from e
    con.execute(sql); con.execute("CHECKPOINT"); con.close()


def ensure_propagation_table(force=False):
    """Step-3 analysis table propagation_features (rebuilt automatically if missing)."""
    _ensure_table("propagation_features", PROPAGATION_SQL, force)


# ---------------------------------------------------------------------------------------------
# T-60 propagation features (Step 4). Prediction moment = 60 minutes before this flight's scheduled departure.
# Every actual predecessor time is put on the absolute axis in the clock of THIS flight's origin airport
# (= predecessor's destination): gate-in = arr_abs; gate-out = arr_abs - elapsed_time (true elapsed time, so
# time-zone shifts are handled). The predecessor's status is read as of T-60; a value is filled only if the event
# happened at or before T-60:
#   landed        gate-in  <= T-60  -> prev_leg_arr_delay, prev_leg_dep_delay known
#   airborne      gate-out <= T-60 < gate-in ("left the gate, not yet at the destination gate"; includes taxi phases)
#                                    -> prev_leg_dep_delay known, prev_leg_arr_delay NaN
#   not_departed  gate-out > T-60   -> both NaN
#   no_prev_leg   first leg of the aircraft's day, or no usable predecessor (cancelled/diverted predecessor,
#                 predecessor not ending at this origin, unknown airport, no tail number)
# Status is explicit, so NaN is never the only signal.  Audit columns (t60_abs, prev_arr_abs, prev_out_abs) are
# stored for the leakage assertion and must never be used as features.
# ---------------------------------------------------------------------------------------------
PROPAGATION_T60_SQL = """
CREATE OR REPLACE TABLE propagation_t60 AS
WITH b AS (
  SELECT f.flight_key, f.tail_number, f.date_key, f.origin_airport_key AS o, f.destination_airport_key AS dst,
         f.scheduled_departure_min AS sdep, f.scheduled_arrival_min AS sarr, f.flight_number,
         (f.cancelled_flag=0 AND f.diverted_flag=0) AS flown, f.departure_delay_min AS dd, f.arrival_delay_min AS ad,
         f.arrival_min, f.elapsed_time_min, f.scheduled_time_min, date_diff('day', DATE '2015-01-01', d.full_date) AS di
  FROM fact_flight f JOIN dim_date d USING(date_key) WHERE f.tail_number IS NOT NULL),
c AS (
  SELECT *, di*1440 + sdep AS sdep_abs,
    CASE WHEN flown THEN di*1440 + arrival_min + 1440*round((sdep + dd + elapsed_time_min - arrival_min)/1440.0) END AS arr_abs,
    CASE WHEN scheduled_time_min IS NOT NULL THEN di*1440 + sarr + 1440*round((sdep + scheduled_time_min - sarr)/1440.0) END AS sarr_abs
  FROM b),
w AS (
  SELECT *, row_number() OVER p AS leg_number_of_day,
    lag(flight_key) OVER p AS prev_key, lag(dst) OVER p AS prev_dst, lag(flown) OVER p AS prev_flown,
    lag(arr_abs) OVER p AS prev_arr_abs, lag(sarr_abs) OVER p AS prev_sarr_abs, lag(ad) OVER p AS prev_ad,
    lag(dd) OVER p AS prev_dd, lag(elapsed_time_min) OVER p AS prev_el
  FROM c WINDOW p AS (PARTITION BY tail_number, date_key ORDER BY sdep, flight_number, flight_key)),
x AS (
  SELECT *, sdep_abs - 60 AS t60_abs, prev_arr_abs - prev_el AS prev_out_abs,
    (prev_key IS NOT NULL AND coalesce(prev_flown, false) AND prev_dst = o AND o <> 0 AND prev_dst <> 0
     AND prev_sarr_abs IS NOT NULL) AS chain_ok
  FROM w),
y AS (
  SELECT *, CASE WHEN NOT chain_ok THEN 'no_prev_leg' WHEN prev_arr_abs <= t60_abs THEN 'landed'
                 WHEN prev_out_abs <= t60_abs THEN 'airborne' ELSE 'not_departed' END AS prev_leg_status
  FROM x)
SELECT flight_key, tail_number, date_key, leg_number_of_day, prev_leg_status,
  CASE WHEN prev_leg_status = 'landed' THEN prev_ad END AS prev_leg_arr_delay,
  CASE WHEN prev_leg_status IN ('landed', 'airborne') THEN prev_dd END AS prev_leg_dep_delay,
  CASE WHEN chain_ok THEN prev_sarr_abs - t60_abs END AS prev_leg_sched_arr_minus_t60,
  CASE WHEN chain_ok THEN sdep_abs - prev_sarr_abs END AS planned_turn_min,
  t60_abs, CASE WHEN chain_ok THEN prev_arr_abs END AS prev_arr_abs, CASE WHEN chain_ok THEN prev_out_abs END AS prev_out_abs,
  sdep_abs
FROM y
"""


def ensure_propagation_t60(force=False):
    """Step-4 T-60 table propagation_t60 (rebuilt automatically if missing)."""
    _ensure_table("propagation_t60", PROPAGATION_T60_SQL, force)
