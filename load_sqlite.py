"""Load the warehouse CSVs into a SQLite database file with PK / FK constraints and indexes.

Run:  .venv/bin/python load_sqlite.py      ->  warehouse/flight_dw.db
"""
import sqlite3
from pathlib import Path

import pandas as pd

WH = Path(__file__).parent / "warehouse"
DB = WH / "flight_dw.db"

DDL = """
CREATE TABLE dim_date (
  date_key INTEGER PRIMARY KEY, full_date TEXT, year INTEGER, quarter INTEGER,
  month_number INTEGER, month_name TEXT, day_of_month INTEGER, day_of_week_number INTEGER,
  day_of_week_name TEXT, week_of_year INTEGER, weekend_flag INTEGER, holiday_flag INTEGER, season TEXT);
CREATE TABLE dim_airline (
  airline_key INTEGER PRIMARY KEY, airline_iata_code TEXT UNIQUE, airline_name TEXT);
CREATE TABLE dim_airport (
  airport_key INTEGER PRIMARY KEY, airport_iata_code TEXT UNIQUE, airport_name TEXT, city TEXT,
  state TEXT, country TEXT, latitude REAL, longitude REAL);
CREATE TABLE dim_route (
  route_key INTEGER PRIMARY KEY, origin_iata_code TEXT, destination_iata_code TEXT,
  route_distance_miles REAL, route_name TEXT, distance_band TEXT, origin_state TEXT,
  destination_state TEXT, interstate_flag INTEGER, UNIQUE (origin_iata_code, destination_iata_code));
CREATE TABLE dim_weather (
  weather_key INTEGER PRIMARY KEY, airport_iata_code TEXT, weather_date TEXT, station_id TEXT,
  station_name TEXT, station_latitude REAL, station_longitude REAL, station_distance_km REAL,
  precipitation_mm REAL, snowfall_mm REAL, snow_depth_mm REAL, tmax_c REAL, tmin_c REAL,
  tavg_c REAL, avg_wind_ms REAL, rain_flag INTEGER, snow_flag INTEGER, high_wind_flag INTEGER,
  weather_severity TEXT);
CREATE TABLE fact_flight (
  flight_key INTEGER PRIMARY KEY,
  date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
  airline_key INTEGER NOT NULL REFERENCES dim_airline(airline_key),
  route_key INTEGER NOT NULL REFERENCES dim_route(route_key),
  origin_airport_key INTEGER NOT NULL REFERENCES dim_airport(airport_key),
  destination_airport_key INTEGER NOT NULL REFERENCES dim_airport(airport_key),
  origin_weather_key INTEGER NOT NULL REFERENCES dim_weather(weather_key),
  destination_weather_key INTEGER NOT NULL REFERENCES dim_weather(weather_key),
  flight_number INTEGER, tail_number TEXT,
  scheduled_departure_min REAL, departure_min REAL, wheels_off_min REAL, wheels_on_min REAL,
  scheduled_arrival_min REAL, arrival_min REAL,
  departure_delay_min REAL, arrival_delay_min REAL, air_time_min REAL, elapsed_time_min REAL,
  scheduled_time_min REAL, taxi_out_min REAL, taxi_in_min REAL, distance_miles REAL,
  air_system_delay_min REAL, security_delay_min REAL, airline_delay_min REAL,
  late_aircraft_delay_min REAL, weather_delay_min REAL,
  cancelled_flag INTEGER, diverted_flag INTEGER, cancellation_reason TEXT,
  departure_delayed_15_flag INTEGER, arrival_delayed_15_flag INTEGER, flight_count INTEGER);
"""
INDEXES = ["date_key", "airline_key", "route_key", "origin_airport_key",
           "destination_airport_key", "origin_weather_key", "destination_weather_key"]


def main():
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(DDL)
    for table in ["dim_date", "dim_airline", "dim_airport", "dim_route", "dim_weather", "fact_flight"]:
        n = 0
        for chunk in pd.read_csv(WH / f"{table}.csv", chunksize=500_000, low_memory=False):
            chunk.to_sql(table, con, if_exists="append", index=False)
            n += len(chunk)
        print(f"loaded {table:<12} {n:>10,} rows")
    for col in INDEXES:
        con.execute(f"CREATE INDEX ix_fact_{col} ON fact_flight({col})")
    con.commit()
    con.execute("PRAGMA foreign_keys = ON")
    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    print(f"foreign_key_check violations: {len(bad)}")
    con.close()
    print(f"database written: {DB}  ({DB.stat().st_size / 1e6:,.0f} MB)")


if __name__ == "__main__":
    main()
