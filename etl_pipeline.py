"""ETL pipeline: raw datasets/ -> star-schema data warehouse (CSV files in warehouse/).

Run:  .venv/bin/python etl_pipeline.py

Tables produced (see DW_TO_BE_BUILT.md):
    dim_date, dim_airline, dim_airport, dim_route, dim_weather, fact_flight
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

DATA = Path(__file__).parent / "datasets"
OUT = Path(__file__).parent / "warehouse"

# Weather settings
WEATHER_ELEMENTS = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "TAVG", "AWND"]
MAX_STATION_KM = 50       # maximum airport-to-station distance
MAX_STATIONS_PER_AIRPORT = 3   # nearest suitable stations used as per-day fallbacks
MIN_PRCP_DAYS = 300       # a "suitable" station reports precipitation on >= this many days
UNKNOWN = 0               # surrogate key of the "Unknown" member in dimensions


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# EXTRACT
# --------------------------------------------------------------------------- #
def extract_flights():
    # hhmm times and codes are read as text so leading zeros / numeric airport codes survive
    dtype = {"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str, "TAIL_NUMBER": str,
             "CANCELLATION_REASON": str}
    return pd.read_csv(DATA / "flights.csv", dtype=dtype, low_memory=False)


def extract_stations():
    # GHCN-Daily fixed-width layout
    cols = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)]
    names = ["station_id", "station_latitude", "station_longitude", "station_elevation",
             "station_state", "station_name"]
    st = pd.read_fwf(DATA / "ghcnd-stations.txt", colspecs=cols, names=names, header=None)
    st["station_name"] = st["station_name"].str.strip()
    return st


def extract_weather(us_station_ids):
    """Stream the 37M-row GHCN file in chunks, keeping only wanted rows."""
    names = ["station_id", "date", "element", "value", "m_flag", "q_flag", "s_flag", "obs_time"]
    keep = []
    reader = pd.read_csv(DATA / "2015.csv", header=None, names=names, chunksize=2_000_000,
                         usecols=["station_id", "date", "element", "value", "q_flag"],
                         dtype={"station_id": str, "date": int, "element": str, "q_flag": str})
    for i, chunk in enumerate(reader, 1):
        chunk = chunk[chunk["element"].isin(WEATHER_ELEMENTS)
                      & chunk["station_id"].isin(us_station_ids)
                      & chunk["q_flag"].isna()]          # blank quality flag = passed QC
        keep.append(chunk[["station_id", "date", "element", "value"]])
        if i % 5 == 0:
            log(f"  weather chunks read: {i}")
    return pd.concat(keep, ignore_index=True)


# --------------------------------------------------------------------------- #
# TRANSFORM: dimensions
# --------------------------------------------------------------------------- #
def build_dim_date():
    d = pd.DataFrame({"full_date": pd.date_range("2015-01-01", "2015-12-31")})
    d["date_key"] = d["full_date"].dt.strftime("%Y%m%d").astype(int)
    d["year"] = d["full_date"].dt.year
    d["quarter"] = d["full_date"].dt.quarter
    d["month_number"] = d["full_date"].dt.month
    d["month_name"] = d["full_date"].dt.month_name()
    d["day_of_month"] = d["full_date"].dt.day
    d["day_of_week_number"] = d["full_date"].dt.dayofweek + 1      # Mon=1 ... Sun=7
    d["day_of_week_name"] = d["full_date"].dt.day_name()
    d["week_of_year"] = d["full_date"].dt.isocalendar().week.astype(int)
    d["weekend_flag"] = (d["day_of_week_number"] >= 6).astype(int)
    holidays = USFederalHolidayCalendar().holidays("2015-01-01", "2015-12-31")
    d["holiday_flag"] = d["full_date"].isin(holidays).astype(int)
    d["season"] = d["month_number"].map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
    return d[["date_key", "full_date", "year", "quarter", "month_number", "month_name",
              "day_of_month", "day_of_week_number", "day_of_week_name", "week_of_year",
              "weekend_flag", "holiday_flag", "season"]]


def build_dim_airline(airlines):
    d = airlines.rename(columns={"IATA_CODE": "airline_iata_code", "AIRLINE": "airline_name"})
    d.insert(0, "airline_key", np.arange(1, len(d) + 1))
    return d


def build_dim_airport(airports):
    d = airports.rename(columns={"IATA_CODE": "airport_iata_code", "AIRPORT": "airport_name",
                                 "CITY": "city", "STATE": "state", "COUNTRY": "country",
                                 "LATITUDE": "latitude", "LONGITUDE": "longitude"})
    d.insert(0, "airport_key", np.arange(1, len(d) + 1))
    unknown = pd.DataFrame([{"airport_key": UNKNOWN, "airport_iata_code": "UNK",
                             "airport_name": "Unknown airport", "city": "Unknown",
                             "state": "UNK", "country": "Unknown",
                             "latitude": np.nan, "longitude": np.nan}])
    return pd.concat([unknown, d], ignore_index=True)


def build_dim_route(flights, dim_airport):
    r = (flights.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], as_index=False)["DISTANCE"]
         .median())
    r.columns = ["origin_iata_code", "destination_iata_code", "route_distance_miles"]
    r.insert(0, "route_key", np.arange(1, len(r) + 1))
    r["route_name"] = r["origin_iata_code"] + "-" + r["destination_iata_code"]
    r["distance_band"] = pd.cut(r["route_distance_miles"], [-1, 500, 1500, np.inf],
                                labels=["Short (<=500)", "Medium (501-1500)", "Long (>1500)"]
                                ).astype(str)
    st = dim_airport.set_index("airport_iata_code")["state"]
    r["origin_state"] = r["origin_iata_code"].map(st).fillna("UNK")
    r["destination_state"] = r["destination_iata_code"].map(st).fillna("UNK")
    r["interstate_flag"] = (r["origin_state"] != r["destination_state"]).astype(int)
    return r


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (broadcasts arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def match_airports_to_stations(dim_airport, stations, prcp_days):
    """Rank the nearest suitable NOAA stations (within MAX_STATION_KM) for every airport."""
    suitable = stations[stations["station_id"].map(prcp_days).fillna(0) >= MIN_PRCP_DAYS] \
        .reset_index(drop=True)
    ap = dim_airport[dim_airport["airport_key"] != UNKNOWN].dropna(subset=["latitude", "longitude"])
    dist = haversine_km(ap["latitude"].to_numpy()[:, None], ap["longitude"].to_numpy()[:, None],
                        suitable["station_latitude"].to_numpy()[None, :],
                        suitable["station_longitude"].to_numpy()[None, :])
    rows = []
    order = np.argsort(dist, axis=1)[:, :MAX_STATIONS_PER_AIRPORT]
    for i, code in enumerate(ap["airport_iata_code"].to_numpy()):
        rank = 0
        for j in order[i]:
            if dist[i, j] <= MAX_STATION_KM:
                rank += 1
                rows.append((code, rank, suitable.at[j, "station_id"], float(dist[i, j])))
    return pd.DataFrame(rows, columns=["airport_iata_code", "station_rank", "station_id",
                                       "station_distance_km"])


def build_dim_weather(weather, stations, airport_station):
    """One row per airport per day; each element comes from the nearest station that has it."""
    wide = (weather.pivot_table(index=["station_id", "date"], columns="element", values="value",
                                aggfunc="first").reset_index())
    for e in WEATHER_ELEMENTS:
        if e not in wide:
            wide[e] = np.nan
    wide["weather_date"] = pd.to_datetime(wide["date"].astype(str), format="%Y%m%d")

    m = airport_station.merge(wide, on="station_id", how="inner").sort_values(
        ["airport_iata_code", "weather_date", "station_rank"])
    g = m.groupby(["airport_iata_code", "weather_date"])
    w = g[WEATHER_ELEMENTS].first()                      # first non-null per element = fallback
    meta = g[["station_id", "station_distance_km"]].first().rename(
        columns={"station_id": "primary_station_id"})
    w = w.join(meta).reset_index()

    # one row for every airport-day that has a station, so flights always find a key
    w = w.merge(stations[["station_id", "station_name", "station_latitude", "station_longitude"]],
                left_on="primary_station_id", right_on="station_id", how="left").drop(
                columns="station_id")

    # unit conversion: GHCN stores tenths of mm / tenths of deg C / tenths of m/s
    out = pd.DataFrame({
        "airport_iata_code": w["airport_iata_code"],
        "weather_date": w["weather_date"],
        "station_id": w["primary_station_id"],
        "station_name": w["station_name"],
        "station_latitude": w["station_latitude"],
        "station_longitude": w["station_longitude"],
        "station_distance_km": w["station_distance_km"].round(2),
        "precipitation_mm": w["PRCP"] / 10,
        "snowfall_mm": w["SNOW"],
        "snow_depth_mm": w["SNWD"],
        "tmax_c": w["TMAX"] / 10,
        "tmin_c": w["TMIN"] / 10,
        "tavg_c": w["TAVG"] / 10,
        "avg_wind_ms": w["AWND"] / 10,
    })
    out["rain_flag"] = (out["precipitation_mm"] > 0).astype(int)
    out["snow_flag"] = ((out["snowfall_mm"] > 0) | (out["snow_depth_mm"] > 0)).astype(int)
    out["high_wind_flag"] = (out["avg_wind_ms"] >= 10).astype(int)
    out["weather_severity"] = np.select(
        [(out["snowfall_mm"] >= 25) | (out["precipitation_mm"] >= 25),
         (out["snow_flag"] == 1) | (out["precipitation_mm"] >= 5) | (out["high_wind_flag"] == 1),
         out["rain_flag"] == 1],
        ["Severe", "Moderate", "Light"], default="Clear")
    out.insert(0, "weather_key", np.arange(1, len(out) + 1))

    unknown = pd.DataFrame([{"weather_key": UNKNOWN, "airport_iata_code": "UNK",
                             "weather_date": pd.NaT, "station_id": "UNK",
                             "weather_severity": "Unknown"}])
    return pd.concat([unknown, out], ignore_index=True)


# --------------------------------------------------------------------------- #
# TRANSFORM: fact
# --------------------------------------------------------------------------- #
def hhmm_to_minutes(s):
    """'0005' / 5 / '2400' -> minutes since midnight (NaN stays NaN)."""
    v = pd.to_numeric(s, errors="coerce")
    return ((v // 100) * 60 + (v % 100)) % 1440


def build_fact(flights, dim_date, dim_airline, dim_airport, dim_route, dim_weather):
    f = pd.DataFrame()
    dt = pd.to_datetime(flights[["YEAR", "MONTH", "DAY"]])
    f["date_key"] = dt.dt.strftime("%Y%m%d").astype(int)

    f["airline_key"] = flights["AIRLINE"].map(
        dim_airline.set_index("airline_iata_code")["airline_key"]).fillna(UNKNOWN).astype(int)

    ap = dim_airport.set_index("airport_iata_code")["airport_key"]
    f["origin_airport_key"] = flights["ORIGIN_AIRPORT"].map(ap).fillna(UNKNOWN).astype(int)
    f["destination_airport_key"] = flights["DESTINATION_AIRPORT"].map(ap).fillna(UNKNOWN).astype(int)

    rt = dim_route.set_index(["origin_iata_code", "destination_iata_code"])["route_key"]
    idx = pd.MultiIndex.from_arrays([flights["ORIGIN_AIRPORT"], flights["DESTINATION_AIRPORT"]])
    f["route_key"] = rt.reindex(idx).to_numpy()

    wk = dim_weather[dim_weather["weather_key"] != UNKNOWN].copy()
    wk["date_key"] = wk["weather_date"].dt.strftime("%Y%m%d").astype(int)
    wmap = wk.set_index(["airport_iata_code", "date_key"])["weather_key"]
    for role, col in (("origin", "ORIGIN_AIRPORT"), ("destination", "DESTINATION_AIRPORT")):
        idx = pd.MultiIndex.from_arrays([flights[col], f["date_key"]])
        f[f"{role}_weather_key"] = wmap.reindex(idx).fillna(UNKNOWN).astype(int).to_numpy()

    # degenerate dimensions
    f["flight_number"] = flights["FLIGHT_NUMBER"]
    f["tail_number"] = flights["TAIL_NUMBER"]

    # times as minutes since midnight
    for src, dst in (("SCHEDULED_DEPARTURE", "scheduled_departure_min"),
                     ("DEPARTURE_TIME", "departure_min"), ("WHEELS_OFF", "wheels_off_min"),
                     ("WHEELS_ON", "wheels_on_min"), ("SCHEDULED_ARRIVAL", "scheduled_arrival_min"),
                     ("ARRIVAL_TIME", "arrival_min")):
        f[dst] = hhmm_to_minutes(flights[src])

    # measures (NaN stays NaN for cancelled / diverted flights, never 0)
    measures = {"DEPARTURE_DELAY": "departure_delay_min", "ARRIVAL_DELAY": "arrival_delay_min",
                "AIR_TIME": "air_time_min", "ELAPSED_TIME": "elapsed_time_min",
                "SCHEDULED_TIME": "scheduled_time_min", "TAXI_OUT": "taxi_out_min",
                "TAXI_IN": "taxi_in_min", "DISTANCE": "distance_miles",
                "AIR_SYSTEM_DELAY": "air_system_delay_min", "SECURITY_DELAY": "security_delay_min",
                "AIRLINE_DELAY": "airline_delay_min", "LATE_AIRCRAFT_DELAY": "late_aircraft_delay_min",
                "WEATHER_DELAY": "weather_delay_min"}
    for src, dst in measures.items():
        f[dst] = flights[src]

    f["cancelled_flag"] = flights["CANCELLED"].astype(int)
    f["diverted_flag"] = flights["DIVERTED"].astype(int)
    f["cancellation_reason"] = flights["CANCELLATION_REASON"]
    f["departure_delayed_15_flag"] = (f["departure_delay_min"] > 15).astype(int)
    f["arrival_delayed_15_flag"] = (f["arrival_delay_min"] > 15).astype(int)
    f["flight_count"] = 1
    f.insert(0, "flight_key", np.arange(1, len(f) + 1))
    return f


# --------------------------------------------------------------------------- #
# LOAD + quality checks
# --------------------------------------------------------------------------- #
def load(tables):
    OUT.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        log(f"  loaded {name:<12} {len(df):>10,} rows -> warehouse/{name}.csv")


def quality_checks(t):
    f = t["fact_flight"]
    log("Quality checks")
    assert f["flight_key"].is_unique, "flight_key not unique"
    for fk, dim, pk in [("date_key", "dim_date", "date_key"),
                        ("airline_key", "dim_airline", "airline_key"),
                        ("route_key", "dim_route", "route_key"),
                        ("origin_airport_key", "dim_airport", "airport_key"),
                        ("destination_airport_key", "dim_airport", "airport_key"),
                        ("origin_weather_key", "dim_weather", "weather_key"),
                        ("destination_weather_key", "dim_weather", "weather_key")]:
        orphans = (~f[fk].isin(t[dim][pk])).sum()
        assert orphans == 0, f"{fk}: {orphans} orphan rows"
        log(f"  FK {fk:<26} OK")
    log(f"  unknown origin airport rows : {(f['origin_airport_key'] == UNKNOWN).sum():,}")
    log(f"  unknown origin weather rows : {(f['origin_weather_key'] == UNKNOWN).sum():,}")


def main():
    t0 = time.time()
    log("EXTRACT")
    flights = extract_flights()
    airlines = pd.read_csv(DATA / "airlines.csv")
    airports = pd.read_csv(DATA / "airports.csv")
    stations = extract_stations()
    us_stations = stations[stations["station_id"].str.startswith("US")]
    log(f"  flights={len(flights):,}  airlines={len(airlines)}  airports={len(airports)}  "
        f"US stations={len(us_stations):,}")
    weather = extract_weather(set(us_stations["station_id"]))
    log(f"  weather rows kept: {len(weather):,}")

    log("TRANSFORM dimensions")
    dim_date = build_dim_date()
    dim_airline = build_dim_airline(airlines)
    dim_airport = build_dim_airport(airports)
    dim_route = build_dim_route(flights, dim_airport)

    prcp_days = weather[weather["element"] == "PRCP"].groupby("station_id")["date"].nunique()
    airport_station = match_airports_to_stations(dim_airport, us_stations, prcp_days)
    log(f"  airports matched to a station: {airport_station['airport_iata_code'].nunique()} "
        f"of {len(airports)}")
    dim_weather = build_dim_weather(weather, stations, airport_station)

    log("TRANSFORM fact")
    fact = build_fact(flights, dim_date, dim_airline, dim_airport, dim_route, dim_weather)

    tables = {"dim_date": dim_date, "dim_airline": dim_airline, "dim_airport": dim_airport,
              "dim_route": dim_route, "dim_weather": dim_weather, "fact_flight": fact}
    quality_checks(tables)
    log("LOAD")
    load(tables)
    log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
