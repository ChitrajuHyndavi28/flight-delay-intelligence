-- Step 4 feature matrix: ONE query over the star schema (warehouse -> mining).
-- Prediction moment = T-60 (60 min before the flight's scheduled departure).  Every column below is known at T-60:
--   schedule / calendar attributes, observed same-day weather (proxy for a forecast), the predecessor's status as of T-60
--   (table propagation_t60, built by common.ensure_propagation_t60), and historical delay rates from the TRAINING period.
-- Population: flown flights (not cancelled, not diverted), October excluded (airport codes unmapped).
-- Target:  is_delayed = arrival_delay_min >= 15   (DOT rule).  `split`: train Jan-Aug, val Sep, test Nov-Dec.
-- Historical rates: expanding window over TRAINING months strictly BEFORE the flight's month (months 2..8), and all
-- 8 training months for val/test rows. A pooled training rate would leak each training row's own label.
-- Smoothed toward the global training rate with m = 20 pseudo-flights.
-- FORBIDDEN here (leakage): departure_delay_min, any actual time of this flight, taxi_out/in, air_time, elapsed_time,
-- the five delay-cause columns, cancellation_reason, arrival_delay_min  (arrival_delay_min is used ONLY to build the target).
WITH base AS (
  SELECT f.flight_key, f.date_key, d.month_number AS month,
    CASE WHEN d.month_number <= 8 THEN 'train' WHEN d.month_number = 9 THEN 'val' ELSE 'test' END AS split,
    (f.arrival_delay_min >= 15)::INTEGER AS is_delayed,
    -- calendar
    d.day_of_week_number AS day_of_week, d.week_of_year, d.weekend_flag, d.holiday_flag, d.season,
    -- schedule
    f.scheduled_departure_min // 60 AS sched_dep_hour, f.scheduled_arrival_min // 60 AS sched_arr_hour,
    l.airline_iata_code AS airline, ao.airport_iata_code AS origin, ad.airport_iata_code AS dest,
    ao.state AS origin_state, ad.state AS dest_state, r.route_name AS route,
    f.distance_miles, r.distance_band, f.scheduled_time_min, r.interstate_flag,
    f.route_key, f.airline_key, f.origin_airport_key,
    -- origin weather (same-day observed; proxy for a forecast)
    wo.precipitation_mm AS o_prcp, wo.snowfall_mm AS o_snow, wo.snow_depth_mm AS o_snwd, wo.tmax_c AS o_tmax, wo.tmin_c AS o_tmin, wo.avg_wind_ms AS o_awnd,
    CASE WHEN wo.weather_key = 0 THEN 'unknown'
         WHEN coalesce(wo.snowfall_mm,0) > 25 OR coalesce(wo.precipitation_mm,0) > 25 OR coalesce(wo.avg_wind_ms,0) > 13.4 THEN 'severe'
         WHEN coalesce(wo.snowfall_mm,0) > 0 OR coalesce(wo.precipitation_mm,0) > 5 OR coalesce(wo.avg_wind_ms,0) > 8.9 THEN 'moderate'
         ELSE 'clear' END AS o_wx_class,
    (wo.weather_key = 0)::INTEGER AS o_weather_missing,
    -- destination weather
    wd.precipitation_mm AS d_prcp, wd.snowfall_mm AS d_snow, wd.snow_depth_mm AS d_snwd, wd.tmax_c AS d_tmax, wd.tmin_c AS d_tmin, wd.avg_wind_ms AS d_awnd,
    CASE WHEN wd.weather_key = 0 THEN 'unknown'
         WHEN coalesce(wd.snowfall_mm,0) > 25 OR coalesce(wd.precipitation_mm,0) > 25 OR coalesce(wd.avg_wind_ms,0) > 13.4 THEN 'severe'
         WHEN coalesce(wd.snowfall_mm,0) > 0 OR coalesce(wd.precipitation_mm,0) > 5 OR coalesce(wd.avg_wind_ms,0) > 8.9 THEN 'moderate'
         ELSE 'clear' END AS d_wx_class,
    (wd.weather_key = 0)::INTEGER AS d_weather_missing,
    -- propagation, state of the predecessor at T-60 (status explicit; NaN is never the only signal)
    coalesce(pt.prev_leg_status, 'no_prev_leg') AS prev_leg_status, pt.prev_leg_arr_delay, pt.prev_leg_dep_delay,
    pt.prev_leg_sched_arr_minus_t60, pt.planned_turn_min, pt.leg_number_of_day,
    least(d.month_number - 1, 8) AS eff_month
  FROM fact_flight f
  JOIN dim_date d USING (date_key) JOIN dim_airline l USING (airline_key)
  JOIN dim_airport ao ON ao.airport_key = f.origin_airport_key JOIN dim_airport ad ON ad.airport_key = f.destination_airport_key
  JOIN dim_route r USING (route_key)
  JOIN dim_weather wo ON wo.weather_key = f.origin_weather_key JOIN dim_weather wd ON wd.weather_key = f.destination_weather_key
  LEFT JOIN propagation_t60 pt ON pt.flight_key = f.flight_key
  WHERE f.cancelled_flag = 0 AND f.diverted_flag = 0 AND d.month_number <> 10 AND f.arrival_delay_min IS NOT NULL
),
tm AS (SELECT month, route_key, airline_key, origin_airport_key, is_delayed FROM base WHERE split = 'train'),
g  AS (SELECT month AS thru, sum(sum(is_delayed)) OVER w * 1.0 / sum(count(*)) OVER w AS g FROM tm GROUP BY month WINDOW w AS (ORDER BY month)),
rc AS (SELECT route_key,          month AS thru, sum(count(*)) OVER w AS n, sum(sum(is_delayed)) OVER w AS k FROM tm GROUP BY route_key, month          WINDOW w AS (PARTITION BY route_key ORDER BY month)),
ac AS (SELECT airline_key,        month AS thru, sum(count(*)) OVER w AS n, sum(sum(is_delayed)) OVER w AS k FROM tm GROUP BY airline_key, month        WINDOW w AS (PARTITION BY airline_key ORDER BY month)),
oc AS (SELECT origin_airport_key, month AS thru, sum(count(*)) OVER w AS n, sum(sum(is_delayed)) OVER w AS k FROM tm GROUP BY origin_airport_key, month WINDOW w AS (PARTITION BY origin_airport_key ORDER BY month))
SELECT b.* EXCLUDE (route_key, airline_key, origin_airport_key, eff_month),
  (rc.k + 20 * g.g) / (rc.n + 20) AS hist_route_rate,
  (ac.k + 20 * g.g) / (ac.n + 20) AS hist_airline_rate,
  (oc.k + 20 * g.g) / (oc.n + 20) AS hist_origin_rate
FROM base b
LEFT JOIN g ON g.thru = b.eff_month
ASOF LEFT JOIN rc ON b.route_key = rc.route_key AND b.eff_month >= rc.thru
ASOF LEFT JOIN ac ON b.airline_key = ac.airline_key AND b.eff_month >= ac.thru
ASOF LEFT JOIN oc ON b.origin_airport_key = oc.origin_airport_key AND b.eff_month >= oc.thru
