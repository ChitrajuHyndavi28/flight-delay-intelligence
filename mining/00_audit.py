"""Step 0 - data audit. Reads flight_dw.duckdb, writes tables/chart to results/audit/."""
import pandas as pd
from common import connect, outdir, set_plot_style, wx_class_sql

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)
con = connect(); out = outdir("audit"); plt = set_plot_style()

# ---- 1. row counts, NULL rates, cancelled/diverted by month --------------------------------
print("=== 1a. Row counts ===")
counts = pd.DataFrame([(t, con.execute(f"select count(*) from {t}").fetchone()[0]) for t in
                       ["fact_flight", "dim_date", "dim_airline", "dim_airport", "dim_route", "dim_weather"]],
                      columns=["table", "rows"])
print(counts.to_string(index=False)); counts.to_csv(out / "row_counts.csv", index=False)

print("\n=== 1b. NULL rate per column ===")
rows = []
for t in ["fact_flight", "dim_weather", "dim_airport", "dim_route"]:
    cols = con.execute(f"describe {t}").df()["column_name"].tolist()
    sel = ", ".join(f"avg(({c} is null)::int) as \"{c}\"" for c in cols)
    r = con.execute(f"select {sel} from {t}").df().iloc[0]
    rows += [(t, c, round(100 * r[c], 2)) for c in cols]
nulls = pd.DataFrame(rows, columns=["table", "column", "null_pct"])
nulls.to_csv(out / "null_rates.csv", index=False)
print(nulls[nulls.null_pct > 0].to_string(index=False))

# NULLs that are structural vs. unexpected: delay/time columns among non-cancelled non-diverted
print("\n--- NULL % among flown flights (not cancelled, not diverted) ---")
flown_cols = ["tail_number", "departure_min", "arrival_min", "departure_delay_min", "arrival_delay_min",
              "air_time_min", "elapsed_time_min", "taxi_out_min", "taxi_in_min", "scheduled_time_min",
              "air_system_delay_min", "late_aircraft_delay_min", "weather_delay_min"]
sel = ", ".join(f"round(100*avg(({c} is null)::int),2) as {c}" for c in flown_cols)
print(con.execute(f"select count(*) flown, {sel} from fact_flight where cancelled_flag=0 and diverted_flag=0")
      .df().T.to_string(header=False))

print("\n--- referential integrity / duplicates ---")
print(con.execute("""select
  (select count(*) - count(distinct flight_key) from fact_flight) dup_flight_keys,
  (select count(*) from fact_flight f left join dim_route r using(route_key) where r.route_key is null) orphan_route,
  (select count(*) from fact_flight f left join dim_date r using(date_key) where r.date_key is null) orphan_date,
  (select count(*) from fact_flight where cancelled_flag=1 and diverted_flag=1) cancelled_and_diverted,
  (select count(*) from fact_flight where cancelled_flag=1 and cancellation_reason is null) canc_no_reason,
  (select sum(flight_count) from fact_flight) sum_flight_count""").df().to_string(index=False))

print("\n=== 1c. Cancelled / diverted by month ===")
monthly = con.execute("""
select d.month_number as month, count(*) flights, sum(cancelled_flag) cancelled, sum(diverted_flag) diverted,
       round(100.0*sum(cancelled_flag)/count(*),2) cancel_pct, round(100.0*sum(diverted_flag)/count(*),2) divert_pct
from fact_flight f join dim_date d using(date_key) group by 1 order by 1""").df()
print(monthly.to_string(index=False)); monthly.to_csv(out / "cancel_divert_by_month.csv", index=False)

# ---- 2. Unknown airport / weather by month -------------------------------------------------
print("\n=== 2. Unknown airport / weather by month ===")
unk = con.execute("""
select d.month_number as month, count(*) flights,
  sum((f.origin_airport_key=0)::int) unk_origin_airport, sum((f.destination_airport_key=0)::int) unk_dest_airport,
  sum((f.origin_weather_key=0)::int) unk_origin_wx, sum((f.destination_weather_key=0)::int) unk_dest_wx,
  round(100.0*avg((f.origin_airport_key=0)::int),1) pct_unk_origin_airport,
  round(100.0*avg((f.origin_weather_key=0)::int),1) pct_unk_origin_wx
from fact_flight f join dim_date d using(date_key) group by 1 order by 1""").df()
print(unk.to_string(index=False)); unk.to_csv(out / "unknown_by_month.csv", index=False)

oct_pct = unk.loc[unk.month == 10, "pct_unk_origin_airport"].iloc[0]
print(f"\nOctober: {oct_pct}% of origins are Unknown -> {'EXCLUDE from airport-level & modeling work' if oct_pct > 50 else 'keep'}")
print("Non-October Unknown-weather origins (airports with no station inside the radius):")
print(con.execute("""select a.airport_iata_code airport, count(*) flights from fact_flight f
  join dim_airport a on a.airport_key=f.origin_airport_key join dim_date d using(date_key)
  where f.origin_weather_key=0 and f.origin_airport_key<>0 and d.month_number<>10
  group by 1 order by 2 desc limit 10""").df().to_string(index=False))
print(con.execute("""select count(*) flights_unknown_wx_nonoct, round(100.0*count(*)/(select count(*) from fact_flight
  join dim_date using(date_key) where month_number<>10),2) pct from fact_flight f join dim_date d using(date_key)
  where f.origin_weather_key=0 and d.month_number<>10""").df().to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(unk.month, unk.pct_unk_origin_airport, color="#c0392b")
ax.set_xticks(range(1, 13)); ax.set_xlabel("Month (2015)"); ax.set_ylabel("% flights with Unknown origin airport")
ax.set_title("Data-quality limitation: October airport codes unmapped (airport_key = 0)")
fig.savefig(out / "unknown_airport_by_month.png"); plt.close(fig)

# ---- 3. weather severity / flags -----------------------------------------------------------
print("\n=== 3. Weather severity & flags ===")
print("Populated? non-null counts in dim_weather (113k station-days):")
print(con.execute("""select count(*) as n_rows, count(weather_severity) severity, count(rain_flag) rain, count(snow_flag) snow,
  count(high_wind_flag) high_wind, count(precipitation_mm) prcp, count(snowfall_mm) snow_mm,
  count(snow_depth_mm) snwd, count(tmax_c) tmax, count(tavg_c) tavg, count(avg_wind_ms) awnd from dim_weather""")
      .df().to_string(index=False))
print("\nWarehouse weather_severity vs. spec-derived class (station-days):")
xt = con.execute(f"""select w.weather_severity as warehouse_label, {wx_class_sql('w')} as derived_class, count(*) station_days
  from dim_weather w group by all order by 1,2""").df()
print(xt.to_string(index=False)); xt.to_csv(out / "weather_label_vs_derived.csv", index=False)

print("\nDistribution of derived class, weighted by flights (origin side, excluding Oct):")
dist = con.execute(f"""select {wx_class_sql('w')} as derived_class, count(*) flights,
  round(100.0*count(*)/sum(count(*)) over (),2) pct
  from fact_flight f join dim_weather w on w.weather_key=f.origin_weather_key join dim_date d using(date_key)
  where d.month_number<>10 group by 1 order by 2 desc""").df()
print(dist.to_string(index=False)); dist.to_csv(out / "weather_class_distribution.csv", index=False)

print("\nStation distance (km) for matched weather rows:")
print(con.execute("select round(min(station_distance_km),2) min, round(median(station_distance_km),2) median, "
                  "round(quantile_cont(station_distance_km,0.95),2) p95, round(max(station_distance_km),2) max "
                  "from dim_weather where weather_key>0").df().to_string(index=False))
