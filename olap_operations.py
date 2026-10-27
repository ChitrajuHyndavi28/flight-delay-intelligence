"""OLAP operations (Roll-Up, Drill-Down, Slice, Dice, Pivot) on the warehouse built by etl_pipeline.py.

Run:  .venv/bin/python olap_operations.py
"""
from pathlib import Path

import pandas as pd

WH = Path(__file__).parent / "warehouse"
pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:,.2f}".format)


def load_cube():
    """Join the fact table to the dimensions it needs (the 'cube' for the examples)."""
    fact = pd.read_csv(WH / "fact_flight.csv", usecols=[
        "date_key", "airline_key", "origin_airport_key", "departure_delay_min",
        "arrival_delay_min", "cancelled_flag", "diverted_flag"])
    date = pd.read_csv(WH / "dim_date.csv", usecols=[
        "date_key", "full_date", "year", "quarter", "month_number", "month_name"])
    airline = pd.read_csv(WH / "dim_airline.csv")
    origin = pd.read_csv(WH / "dim_airport.csv", usecols=["airport_key", "state"]).rename(
        columns={"airport_key": "origin_airport_key", "state": "origin_state"})
    cube = (fact.merge(date, on="date_key").merge(airline, on="airline_key")
            .merge(origin, on="origin_airport_key"))
    cube["quarter"] = "Q" + cube["quarter"].astype(str)
    # on-time = arrived within 15 minutes; only flights that actually arrived are eligible
    arrived = (cube["cancelled_flag"] == 0) & (cube["diverted_flag"] == 0)
    cube["arrived"] = arrived.astype(int)
    cube["on_time"] = (arrived & (cube["arrival_delay_min"] <= 15)).astype(int)
    return cube


def summarize(df, by):
    """Aggregate measures at the level given by `by` (always recomputed from base rows)."""
    g = df.groupby(by, observed=True)
    out = g.agg(flight_count=("date_key", "size"),
                avg_departure_delay=("departure_delay_min", "mean"),
                avg_arrival_delay=("arrival_delay_min", "mean"),
                cancellation_count=("cancelled_flag", "sum"))
    out["cancellation_rate_pct"] = out["cancellation_count"] / out["flight_count"] * 100
    return out


def section(title, why):
    print(f"\n{'=' * 90}\n{title}\n{why}\n{'=' * 90}")


# --------------------------------------------------------------------------- #
def roll_up(cube):
    section("1. ROLL-UP  Day -> Month -> Quarter -> Year",
            "Moves UP the DIM_DATE hierarchy: group-by keys are removed and measures re-aggregated.")
    cols = ["flight_count", "avg_departure_delay", "avg_arrival_delay", "cancellation_count"]
    levels = {"Day": ["full_date"],
              "Month": ["year", "month_number"],
              "Quarter": ["year", "quarter"],
              "Year": ["year"]}
    for name, by in levels.items():
        res = summarize(cube, by)[cols]
        print(f"\n-- Level: {name}  ({len(res)} rows, first 5 shown)")
        print(res.head(5))


def drill_down(cube, quarter="Q1", month=1):
    section("2. DRILL-DOWN  Year -> Quarter -> Month -> Day  (2015)",
            "Moves DOWN the same hierarchy: start at Year=2015 and add one lower level, "
            "fixing the parent value chosen at the previous step.")
    cols = ["flight_count", "avg_arrival_delay", "cancellation_count"]
    d = cube[cube["year"] == 2015]
    print("\n-- Step 1: Year = 2015")
    print(summarize(d, ["year"])[cols])
    print("\n-- Step 2: + Quarter")
    print(summarize(d, ["year", "quarter"])[cols])
    q = d[d["quarter"] == quarter]
    print(f"\n-- Step 3: + Month (within {quarter})")
    print(summarize(q, ["quarter", "month_number"])[cols])
    m = q[q["month_number"] == month]
    print(f"\n-- Step 4: + Day (within month {month}, first 7 days shown)")
    print(summarize(m, ["full_date"])[cols].head(7))


def slice_op(cube, airline="Delta Air Lines Inc."):
    section(f"3. SLICE  Airline = {airline}",
            "ONE condition on ONE dimension (DIM_AIRLINE); every other dimension stays free.")
    s = cube[cube["airline_name"] == airline]
    res = pd.Series({
        "total_flights": len(s),
        "avg_departure_delay": s["departure_delay_min"].mean(),
        "avg_arrival_delay": s["arrival_delay_min"].mean(),
        "cancellation_rate_pct": s["cancelled_flag"].mean() * 100,
        "on_time_pct": s["on_time"].sum() / s["arrived"].sum() * 100})
    print(res.to_string())


def dice_op(cube, airline="Delta Air Lines Inc."):
    section(f"4. DICE  Year=2015, Quarter=Q1, Airline={airline}, Origin State=NY",
            "Conditions on SEVERAL dimensions at once (DIM_DATE, DIM_AIRLINE, DIM_AIRPORT-origin) "
            "-> a smaller sub-cube.")
    d = cube[(cube["year"] == 2015) & (cube["quarter"] == "Q1")
             & (cube["airline_name"] == airline) & (cube["origin_state"] == "NY")]
    res = pd.Series({
        "flight_count": len(d),
        "avg_departure_delay": d["departure_delay_min"].mean(),
        "avg_arrival_delay": d["arrival_delay_min"].mean(),
        "cancellation_rate_pct": d["cancelled_flag"].mean() * 100})
    print(res.to_string())


def pivot_op(cube):
    section("5. PIVOT  Airline (rows) x Month (columns) -> Average Arrival Delay",
            "Rotates the axes of the cube to show two dimensions side by side; no filtering, "
            "no change of aggregation level.")
    p = cube[cube["year"] == 2015].pivot_table(
        index="airline_name", columns="month_number", values="arrival_delay_min", aggfunc="mean")
    p.columns = [pd.Timestamp(2015, m, 1).strftime("%b") for m in p.columns]
    print(p)
    print("\nWorst month per airline:\n" + p.idxmax(axis=1).to_string())
    print("\nWorst airline per month:\n" + p.idxmax(axis=0).to_string())


def main():
    cube = load_cube()
    print(f"Cube loaded: {len(cube):,} flights")
    roll_up(cube)
    drill_down(cube)
    slice_op(cube)
    dice_op(cube)
    pivot_op(cube)


if __name__ == "__main__":
    main()
