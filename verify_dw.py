"""Independent verification of the warehouse and OLAP results against the RAW files.

Run:  .venv/bin/python verify_dw.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW, WH = ROOT / "datasets", ROOT / "warehouse"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


raw = pd.read_csv(RAW / "flights.csv", dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str},
                  low_memory=False)
fact = pd.read_csv(WH / "fact_flight.csv", low_memory=False)
d_date = pd.read_csv(WH / "dim_date.csv", parse_dates=["full_date"])
d_air = pd.read_csv(WH / "dim_airline.csv")
d_ap = pd.read_csv(WH / "dim_airport.csv")
d_rt = pd.read_csv(WH / "dim_route.csv")
d_w = pd.read_csv(WH / "dim_weather.csv", parse_dates=["weather_date"])

print("\n--- A. Fact vs raw flights.csv ---")
check("row count equal", len(fact) == len(raw), f"{len(fact):,} vs {len(raw):,}")
check("flight_key unique", fact["flight_key"].is_unique)
for src, dst in [("DEPARTURE_DELAY", "departure_delay_min"), ("ARRIVAL_DELAY", "arrival_delay_min"),
                 ("AIR_TIME", "air_time_min"), ("ELAPSED_TIME", "elapsed_time_min"),
                 ("TAXI_OUT", "taxi_out_min"), ("TAXI_IN", "taxi_in_min"),
                 ("DISTANCE", "distance_miles"), ("WEATHER_DELAY", "weather_delay_min"),
                 ("LATE_AIRCRAFT_DELAY", "late_aircraft_delay_min")]:
    same_sum = np.isclose(raw[src].sum(), fact[dst].sum())
    same_nulls = raw[src].isna().sum() == fact[dst].isna().sum()
    check(f"{dst}: sum and NULL count match raw", same_sum and same_nulls,
          f"sum={fact[dst].sum():,.0f}, nulls={fact[dst].isna().sum():,}")
check("cancelled count", raw["CANCELLED"].sum() == fact["cancelled_flag"].sum(),
      f"{fact['cancelled_flag'].sum():,}")
check("diverted count", raw["DIVERTED"].sum() == fact["diverted_flag"].sum(),
      f"{fact['diverted_flag'].sum():,}")
check("no 0-filled delays for cancelled flights",
      fact.loc[fact["cancelled_flag"] == 1, "arrival_delay_min"].isna().all())

# row-by-row alignment: flight_key i  <->  raw row i-1
rng = np.random.default_rng(1)
pos = rng.choice(len(raw), 2000, replace=False)
r, f = raw.iloc[pos].reset_index(drop=True), fact.iloc[pos].reset_index(drop=True)
check("flight_number aligned (2,000 random rows)", (r["FLIGHT_NUMBER"] == f["flight_number"]).all())
check("tail_number aligned", (r["TAIL_NUMBER"].fillna("") == f["tail_number"].fillna("")).all())
check("date_key aligned",
      (pd.to_datetime(r[["YEAR", "MONTH", "DAY"]]).dt.strftime("%Y%m%d").astype(int)
       == f["date_key"]).all())
code = d_air.set_index("airline_key")["airline_iata_code"]
check("airline_key resolves to raw AIRLINE", (f["airline_key"].map(code) == r["AIRLINE"]).all())
ap = d_ap.set_index("airport_key")["airport_iata_code"]
known = f["origin_airport_key"] != 0
check("origin_airport_key resolves to raw ORIGIN_AIRPORT (known airports)",
      (f.loc[known, "origin_airport_key"].map(ap) == r.loc[known, "ORIGIN_AIRPORT"]).all())
check("destination_airport_key resolves to raw DESTINATION_AIRPORT",
      (f.loc[f["destination_airport_key"] != 0, "destination_airport_key"].map(ap)
       == r.loc[f["destination_airport_key"] != 0, "DESTINATION_AIRPORT"]).all())
rt = d_rt.set_index("route_key")
check("route_key resolves to raw origin/destination",
      ((f["route_key"].map(rt["origin_iata_code"]) == r["ORIGIN_AIRPORT"])
       & (f["route_key"].map(rt["destination_iata_code"]) == r["DESTINATION_AIRPORT"])).all())
# hhmm -> minutes: 0005 -> 5, 2400 -> 0, 1330 -> 810
sd = pd.to_numeric(r["SCHEDULED_DEPARTURE"], errors="coerce")
check("scheduled_departure_min = hh*60+mm", (f["scheduled_departure_min"] == ((sd // 100) * 60 + sd % 100) % 1440).all())

print("\n--- B. Dimensions ---")
check("dim_date has 365 unique days of 2015", d_date["date_key"].is_unique and len(d_date) == 365)
check("2015-01-01 is Thursday (day_of_week 4)",
      d_date.loc[d_date["date_key"] == 20150101, "day_of_week_name"].iloc[0] == "Thursday")
check("raw DAY_OF_WEEK (1=Mon) agrees with dim_date for every flight date",
      (raw[["YEAR", "MONTH", "DAY", "DAY_OF_WEEK"]].drop_duplicates()
       .assign(date_key=lambda x: pd.to_datetime(x[["YEAR", "MONTH", "DAY"]]).dt.strftime("%Y%m%d").astype(int))
       .merge(d_date[["date_key", "day_of_week_number"]], on="date_key")
       .eval("DAY_OF_WEEK == day_of_week_number").all()))
check("dim_date quarter counts 90/91/92/92", d_date.groupby("quarter").size().tolist() == [90, 91, 92, 92])
check("holiday_flag: 10 US federal holidays in 2015", d_date["holiday_flag"].sum() == 10,
      f"found {d_date['holiday_flag'].sum()}")
air_raw = pd.read_csv(RAW / "airlines.csv")
check("dim_airline equals airlines.csv", set(air_raw["IATA_CODE"]) == set(d_air["airline_iata_code"]))
ap_raw = pd.read_csv(RAW / "airports.csv")
check("dim_airport = airports.csv + 1 Unknown", len(d_ap) == len(ap_raw) + 1 and (d_ap["airport_key"] == 0).sum() == 1)
pairs = raw.drop_duplicates(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]).shape[0]
check("dim_route = distinct raw origin/destination pairs", len(d_rt) == pairs, f"{pairs:,}")
dist_var = raw.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["DISTANCE"].nunique().max()
spread = raw.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["DISTANCE"].agg(lambda s: s.max() - s.min()).max()
# source has 1-2 mile rounding noise on 68 routes; the fact keeps each flight's exact distance
check("route distance varies by <= 2 miles within a route in raw", spread <= 2, f"max spread={spread}")

print("\n--- C. Weather ---")
w = d_w[d_w["weather_key"] != 0]
check("dim_weather: one row per airport per day", not w.duplicated(["airport_iata_code", "weather_date"]).any())
check("all airport-station distances <= 50 km", w["station_distance_km"].max() <= 50,
      f"max={w['station_distance_km'].max():.1f}")


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742.0176 * np.arcsin(np.sqrt(a))   # 2 * mean radius, independent re-implementation


ws = w.drop_duplicates("airport_iata_code").merge(
    ap_raw, left_on="airport_iata_code", right_on="IATA_CODE")
calc = hav(ws["LATITUDE"], ws["LONGITUDE"], ws["station_latitude"], ws["station_longitude"])
check("station_distance_km matches independent haversine", np.allclose(calc, ws["station_distance_km"], atol=0.05))

# spot-check actual values against the raw NOAA file for 3 airport-days
stn = pd.read_csv(RAW / "2015.csv", header=None, usecols=[0, 1, 2, 3, 5],
                  names=["id", "date", "el", "val", "q"], dtype={"id": str, "el": str, "q": str},
                  chunksize=3_000_000)
need = {}
sample = w.sample(5, random_state=3)
for _, s in sample.iterrows():
    need[(s["station_id"], int(s["weather_date"].strftime("%Y%m%d")))] = s
ids = {k[0] for k in need}
hits = pd.concat([c[c["id"].isin(ids) & c["q"].isna() & c["el"].isin(["PRCP", "TMAX", "TMIN"])] for c in stn])
ok_all = True
for (sid, dt), s in need.items():
    h = hits[(hits["id"] == sid) & (hits["date"] == dt)].set_index("el")["val"]
    for el, col, div in (("PRCP", "precipitation_mm", 10), ("TMAX", "tmax_c", 10), ("TMIN", "tmin_c", 10)):
        if el in h.index and pd.notna(s[col]):
            # the dim may use a fallback station if the primary lacked the element, so only
            # compare when the primary station has the value
            ok_all &= np.isclose(h[el] / div, s[col])
check("PRCP/TMAX/TMIN of 5 random airport-days equal raw NOAA values (unit-converted)", ok_all)
check("fact weather keys: origin weather date and airport match the flight",
      (lambda m: ((m["airport_iata_code"] == m["ORIGIN_AIRPORT"]) &
                  (m["weather_date"].dt.strftime("%Y%m%d").astype(int) == m["date_key"])).all())(
          fact.iloc[pos].reset_index(drop=True)[["origin_weather_key", "date_key"]].assign(
              ORIGIN_AIRPORT=r["ORIGIN_AIRPORT"])
          .merge(d_w, left_on="origin_weather_key", right_on="weather_key")
          .query("weather_key != 0")))
unk_airports = raw.loc[~raw["ORIGIN_AIRPORT"].isin(ap_raw["IATA_CODE"]), "ORIGIN_AIRPORT"]
check("all numeric/unknown origin codes map to Unknown member",
      (fact.loc[~raw["ORIGIN_AIRPORT"].isin(ap_raw["IATA_CODE"]), "origin_airport_key"] == 0).all(),
      f"{len(unk_airports):,} rows, sample codes {sorted(unk_airports.unique())[:3]}")

print("\n--- D. OLAP numbers recomputed from RAW (no warehouse tables used) ---")
dl = raw[raw["AIRLINE"] == "DL"]
arrived = dl[(dl["CANCELLED"] == 0) & (dl["DIVERTED"] == 0)]
exp = {"total": len(dl), "dep": dl["DEPARTURE_DELAY"].mean(), "arr": dl["ARRIVAL_DELAY"].mean(),
       "canc": dl["CANCELLED"].mean() * 100, "ontime": (arrived["ARRIVAL_DELAY"] <= 15).mean() * 100}
print("   expected Delta slice:", {k: round(v, 2) for k, v in exp.items()})
check("Slice: Delta total flights = 875,881", exp["total"] == 875_881)
check("Slice: Delta avg dep delay = 7.37", round(exp["dep"], 2) == 7.37)
check("Slice: Delta avg arr delay = 0.19", round(exp["arr"], 2) == 0.19)
check("Slice: Delta cancellation rate = 0.44%", round(exp["canc"], 2) == 0.44)
check("Slice: Delta on-time % = 87.00", round(exp["ontime"], 2) == 87.00)

ny = set(ap_raw.loc[ap_raw["STATE"] == "NY", "IATA_CODE"])
dd = dl[(dl["MONTH"] <= 3) & dl["ORIGIN_AIRPORT"].isin(ny)]
check("Dice: Delta Q1 NY flight count = 14,617", len(dd) == 14_617, f"{len(dd):,}")
check("Dice: avg dep delay = 21.60", round(dd["DEPARTURE_DELAY"].mean(), 2) == 21.60)
check("Dice: avg arr delay = 12.65", round(dd["ARRIVAL_DELAY"].mean(), 2) == 12.65)
check("Dice: cancellation rate = 4.89%", round(dd["CANCELLED"].mean() * 100, 2) == 4.89)

check("Roll-up Year: 5,819,079 flights / 89,884 cancellations",
      len(raw) == 5_819_079 and raw["CANCELLED"].sum() == 89_884)
q1 = raw[raw["MONTH"] <= 3]
check("Roll-up Q1: flights 1,403,471, avg arr delay 6.24",
      len(q1) == 1_403_471 and round(q1["ARRIVAL_DELAY"].mean(), 2) == 6.24)
check("Roll-up: quarter flights sum to year", sum(len(raw[raw['MONTH'].between(a, a + 2)]) for a in (1, 4, 7, 10)) == len(raw))
jan1 = raw[(raw["MONTH"] == 1) & (raw["DAY"] == 1)]
check("Drill-down day 2015-01-01: 13,950 flights, 466 cancelled",
      len(jan1) == 13_950 and jan1["CANCELLED"].sum() == 466)
b6feb = raw[(raw["AIRLINE"] == "B6") & (raw["MONTH"] == 2)]["ARRIVAL_DELAY"].mean()
check("Pivot cell JetBlue/Feb = 18.66", round(b6feb, 2) == 18.66, f"{b6feb:.2f}")
us = raw[raw["AIRLINE"] == "US"].groupby("MONTH")["ARRIVAL_DELAY"].count()
check("Pivot: US Airways has no arrival data after June (matches raw)", us.index.max() == 6,
      f"last month in raw = {us.index.max()}")

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
