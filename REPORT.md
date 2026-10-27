# Flight Delay & Airport Operations Intelligence — Analysis & Mining Report

> Sections 1-10 complete (9 = neural network, ensemble, PSO tuning, weather impact and base-paper comparison; 10 = airport clustering and cancellation association rules). Still to come: the Key Insights list.

## 1. Problem statement and objectives
Airline delays cost passengers and carriers time and money, and their drivers (weather, airline operations, congestion, aircraft rotation) are routinely conflated. This project builds an analysis and mining layer on a star-schema data warehouse of the 2015 US DOT on-time dataset (5,819,079 flight legs) joined to NOAA GHCN-Daily weather.

| ID | Objective |
|---|---|
| O1 | Dimensional model and ETL for flights + weather (built before this phase; `etl_pipeline.py`, `DW_TO_BE_BUILT.md`) |
| O2 | OLAP analysis: roll-up, drill-down, slice, dice, pivot; export results for slides |
| O3 | Root-cause and weather-impact quantification |
| O4 | Delay propagation along an aircraft's day (via `tail_number`) |
| O5 | Predictive mining: LightGBM vs baselines, with explainability |
| O6 | Pattern discovery: airport clustering and cancellation association rules |

## 2. Pipeline
| Step | Script | Output |
|---|---|---|
| Build | `mining/build_duckdb.py` | Loads `warehouse/*.csv` into `flight_dw.duckdb` (5,819,079 fact rows). Run first. |
| 0 | `mining/00_audit.py` | `results/audit/` |
| 1 | `mining/01_olap_export.py` | `results/olap/` |
| 2 | `mining/02_root_cause_weather.py` | `results/root_cause_weather/` |
| 3 | `mining/03_delay_propagation.py` | `results/propagation/`; table `propagation_features` in `flight_dw.duckdb` (+ parquet) |
| 4 | `mining/04a_t60_spot_check.py`, `mining/04_features.sql`, `mining/04_train_compare.py` (driver `mining/run_step4.sh`), `mining/04_explain.py` | `results/model_comparison/`, `results/explain/`; tables `propagation_t60` in `flight_dw.duckdb`; cache `mining_cache/` |

Shared helpers (DB path, plot style, weather-class SQL, seed = 42, DuckDB capped at 6 GB / 6 threads): `mining/common.py`. Environment: Python 3.9 venv; LightGBM/XGBoost need `brew install libomp` on macOS. Every script runs on its own and reads from `flight_dw.duckdb`.

## 3. Definitions and decisions (apply to every section)
- **Delayed = `arrival_delay_min >= 15`** (DOT rule; on-time = < 15 min). This aligns with DOT published figures and with the DOT delay-cause columns, which are filled for delays >= 15 min. *The warehouse column `arrival_delayed_15_flag` uses `> 15` and is left unchanged; the analysis does not use it.* Switching from `> 15` to `>= 15` lowered on-time % by 0.55–0.92 pp in every cut (overall 82.09% -> 81.39%, -0.70 pp); no ranking changed.
- **Flown flights** = not cancelled and not diverted. Delay metrics use flown flights; cancellation rate uses all scheduled flights.
- **Departure-hour buckets (scheduled departure):** Morning 05:00–11:59, Afternoon 12:00–17:59, Evening 18:00–21:59, Night 22:00–04:59.
- **Weather severity (derived, `wx_class_sql()` in `common.py`).** The spec's thresholds, converted to the warehouse's units (mm, m/s):

  | Class | Rule (any condition) | Spec units (tenths) | Warehouse units |
  |---|---|---|---|
  | severe | snowfall | > 25 mm | > 25 mm |
  | | precipitation | > 250 tenths-mm | > 25 mm |
  | | avg wind | > 134 tenths m/s | > 13.4 m/s |
  | moderate | snowfall | > 0 | > 0 mm |
  | | precipitation | > 50 tenths-mm | > 5 mm |
  | | avg wind | > 89 tenths m/s | > 8.9 m/s |
  | clear | otherwise | | |
  | unknown | `weather_key = 0` (no station in radius / unmapped airport) | | |

  NULL measurements are treated as 0. The warehouse's own 4-level `weather_severity` (Clear/Light/Moderate/Severe, undocumented rule) disagrees with it on 7,500 of 112,786 station-days (6,903 warehouse "Moderate" that are "clear" here, 300 "Severe" that are "moderate", 292 "Clear/Light" that are "moderate", 5 "Moderate" that are "severe"), and its "Light" tier (16,247 station-days, rain < 5 mm) is merged into clear here. It is used only as a sensitivity check (Section 6.4).
- **October 2015.** The source uses numeric BTS airport IDs, which the ETL mapped to the Unknown airport/weather member (`airport_key = 0`) **by design** (100% of October flights). October is excluded from airport-level, weather, propagation and modeling work and kept in time-only OLAP (month/weekday/hour/airline). *Future work:* map BTS airport ID -> IATA with the BTS lookup table and reload.
- **Split for modeling:** train Jan–Aug, validation Sep, test Nov–Dec.
- **Unknown-weather airports** (SJU, STT, ECP, BQN, STX, PSE, DVL, GUM...; 0.71% of non-October flights): excluded from weather comparisons; kept in modeling with weather = NaN plus an `is_weather_missing` indicator.
- **Missing values in modeling:** `tavg_c` dropped (98% NULL). LightGBM/XGBoost use native NaN; other models use median imputation + missing-indicator columns, fitted on the training set only (sklearn Pipeline).
- **Delay-cause shares** are computed over flights with `arrival_delay_min >= 15`, the only flights where DOT fills the cause columns (verified: all 1,063,439 such flights have all five causes, summing exactly to arrival delay; flights < 15 min have none).
- **Cancellation share (one definition).** "Weather = X% of cancellations" is reported over **all 12 months: 48,851 / 89,884 = 54.4%**. Airport-level analyses exclude October: 47,874 / 87,430 = 54.8%. The Step 2c weather-comparison scope: 47,691 / 87,076 = 54.8%. (`results/root_cause_weather/2b_cancellation_share_reconciliation.csv`.) *Correction:* an earlier interim summary quoted 105,813 total cancellations (1.82%); the correct total is 89,884 (1.54% of flights).
- **US Airways (US) covers January–June only** (merged into American Airlines). Its annual rankings reflect half a year.
- **Weather is observed same-day weather**, used as a proxy for a forecast in the models.

## 4. Data audit and limitations (Step 0)
**Method.** Row counts, NULL rates, referential checks, cancelled/diverted counts by month, Unknown airport/weather counts by month, and a comparison of the warehouse severity label with the derived class. Files: `results/audit/`.

**Key facts**
1. 5,819,079 flights; no duplicate keys, no orphan foreign keys, no flight both cancelled and diverted; every cancelled flight has a reason.
2. 89,884 flights cancelled (1.54%) and 15,187 diverted (0.26%). Cancellations peak in February (4.78%) and are lowest in September (0.45%).
3. NULLs are structural: the five cause columns are 81.7% NULL (filled only for delays >= 15 min); among flown flights all time/delay columns are complete; `tail_number` is NULL for 0.25% of flights.
4. **October is 100% Unknown** (486,165 flights): airport keys and weather keys are all 0 (see Section 3).
5. Outside October, 37,779 flights (0.71%) have Unknown weather because the airport has no GHCN station inside the radius (SJU 24,656; STT 4,348; ECP 4,169; BQN 1,343; STX 939 ...).
6. Weather class distribution (origin, flights, excl. October): clear 83.2%, moderate 11.8%, severe 4.3%, unknown 0.7%.

**Limitations**
- October exclusion removes 8.4% of flights from airport-level, weather, propagation and modeling work.
- One weather station per airport (median 0.85 km away, 95th percentile 5.6 km, maximum 49.8 km); a single point measurement at the airport's nearest station.
- Weather is same-day observed, not a forecast; destination and en-route weather are not in the weather comparisons.
- `snow_depth_mm` 20% NULL, `snowfall_mm` 9.6% NULL, `avg_wind_ms` 8.2% NULL; NULL treated as "no weather" in the derived class (conservative).

## 5. O2 — OLAP results (Step 1)
**Method.** SQL aggregations over the star schema: on-time %, average arrival/departure delay and cancellation rate by month, day of week, departure hour and hour bucket; airport ranking (>= 5,000 flown departures, October excluded: 113 airports); airline ranking; airline x month heatmap. Time-only cuts use all 12 months. Files: `results/olap/` (CSV + PNG per cut).

**Key table — by departure bucket** (`by_dep_bucket.csv`)

| Bucket | Flights | On-time % | Avg arrival delay (min) | Avg departure delay (min) | Cancel % |
|---|---|---|---|---|---|
| Morning 05–11 | 2,381,083 | 87.20 | 0.00 | 4.77 | 1.41 |
| Afternoon 12–17 | 2,141,110 | 78.88 | 6.57 | 11.37 | 1.50 |
| Evening 18–21 | 1,112,617 | 74.16 | 9.62 | 14.78 | 1.94 |
| Night 22–04 | 184,269 | 78.74 | 4.91 | 10.66 | 1.47 |

(Morning average *arrival* delay is 0.00 min while average *departure* delay is 4.77 min: aircraft recover time in the air against padded schedules. They are different quantities, not a conflict.)

**Findings**
1. **Delay builds through the day.** On-time falls from 92.7% for 05:00 departures to 73.7% at 20:00; evening departures are 2.4x as likely to leave 15+ min late as morning departures (27.0% vs 11.3%).
2. **The evening effect is real but small in rank terms.** Mann-Whitney U on departure delay (evening n = 1,088,433 vs morning n = 2,341,610): p underflows to 0 and is uninformative at this n. Effect sizes: Cliff's delta = **0.225** (exact, 2U/(n1·n2) − 1 on the full data; full-size bootstrap 95% CI 0.224–0.226), a *small* effect (Romano: 0.147 small, 0.33 medium); P(evening flight departs later than morning flight) = 61.2%; mean difference +10.0 min, median difference 3 min (0 vs −3).
3. **Seasonality:** worst months are February (76.65% on-time, 4.78% cancelled) and June (76.52%, 9.6 min average arrival delay); best are September (87.00%) and October (87.56%).
4. **Day of week:** Saturday is best (84.01% on-time, 1.85 min average arrival delay); Thursday is worst on-time (79.99%); Monday has the highest cancellation rate (2.43%).
5. **Airports and airlines:** worst average departure delay: ORD 14.03, EWR 13.55, BWI 13.26, LGA 13.17 min; best: ITO −0.89, LIH 0.46, KOA 1.35, HNL 2.62. Airline on-time %: Hawaiian 88.67, Alaska 86.96, Delta 86.44 ... Frontier 73.84, Spirit 70.29. American Eagle (MQ) cancels 5.1% of flights, 3x the 1.54% network rate. Airline x month peak: Spirit in June, 35.6 min average arrival delay.

**Limitations.** Bucket/hour effects are not adjusted for airline or airport mix (adjusted effects appear in the Step 2 and Step 4 models); airline rankings include different route networks; US Airways is half-year only.

## 6. O3 — Root cause and weather impact (Step 2)

### 6.1 Cause attribution (flights with arrival delay >= 15 min, 1,063,439 flights)
| Cause | Delay minutes | Share |
|---|---|---|
| Late aircraft | 24,961,931 | 39.8% |
| Airline | 20,172,956 | 32.2% |
| Air system (NAS) | 14,335,762 | 22.9% |
| Weather | 3,100,233 | 4.9% |
| Security | 80,985 | 0.1% |

By airline, month and the 20 busiest airports: `2a_cause_share_by_*.csv/png`. Southwest is 51% late-aircraft; Hawaiian 58% airline-caused; Spirit 44% NAS; weather share of delay minutes ranges from 10.1% (ORD) to 1.5% (LAX) among the 20 busiest airports, and peaks in February (7.3%).

### 6.2 Cancellation reasons
Weather accounts for **54.4% of cancellations (48,851 / 89,884)**, airline 28.1%, NAS 17.5%, security 0.02%. Weather-coded cancellations dominate winter: February 15,447 of 20,517 (75%); by airport, weather share ranges from 75% (DFW) and 74% (MDW) to 37% (EWR, mostly NAS-coded) and 28% (LAX).

### 6.3 Weather effect, controlled
**Method.** Origin weather class (clear / moderate / severe, derived; unknown excluded; October excluded). *Raw gap* = pooled class rate minus pooled clear rate. *Controlled gap* = the same difference computed **within each (origin airport, month) cell** that has >= 50 flown flights in both classes, averaged with weights equal to the treated class's flight count. 95% CIs: cluster bootstrap over airports (1,000 resamples, seed 42). *Disrupted* = delayed (>= 15 min) OR cancelled OR diverted (rate over all scheduled flights). A logistic regression `outcome ~ weather + airport + month + hour + airline` (1,000,000-row random sample, standard errors clustered by airport-day) gives adjusted odds ratios.

**Key table — severe vs clear / moderate vs clear** (`2c_raw_vs_controlled_gaps_derived.csv`)

| Metric | Class | Rate (class vs clear) | Raw gap | Controlled gap [95% CI] |
|---|---|---|---|---|
| Delay rate, flown (pp) | severe | 40.4 vs 17.0 | +23.3 | **+23.4 [19.9, 26.5]** |
| | moderate | 27.7 vs 17.0 | +10.6 | +10.7 [9.3, 12.0] |
| Avg arrival delay (min) | severe | 28.1 vs 2.8 | +25.3 | **+25.9 [21.6, 30.3]** |
| | moderate | 12.5 vs 2.8 | +9.7 | +10.1 [8.7, 11.4] |
| Cancellation rate (pp) | severe | 9.0 vs 1.1 | +7.9 | **+7.7 [4.9, 10.7]** |
| | moderate | 3.1 vs 1.1 | +2.0 | +1.8 [1.2, 2.5] |
| Disrupted rate (pp) | severe | 45.9 vs 18.1 | +27.8 | **+27.5 [23.4, 31.3]** |
| | moderate | 30.1 vs 18.1 | +12.0 | +11.9 [10.2, 13.4] |

Adjusted odds ratios vs clear (`2c_logistic_odds_ratios.csv`):

| Outcome | Moderate (OR, 95% CI) | Severe (OR, 95% CI) |
|---|---|---|
| Delayed (arrival >= 15 min) | 1.98 (1.90–2.05) | 3.60 (3.35–3.87) |
| Cancelled | 2.55 (2.26–2.88) | 7.70 (6.57–9.03) |

**Findings**
1. **Severe origin weather raises the delay rate by ~23 pp (40.4% vs 17.0%), average arrival delay by ~26 min, and the cancellation rate ~8x (9.0% vs 1.1%).** Moderate weather has roughly half the delay effect and ~3x the cancellation rate.
2. **Controlling for airport and month did not shrink the gap** (raw +23.3 pp vs controlled +23.4 pp; +25.3 vs +25.9 min), contrary to the common expectation that the raw gap is inflated by weather concentrating in bad-baseline airports/months. Interpretation: within the data, severe days occur across a broad mix of airports and months, so airport/season confounding is small relative to the weather effect itself. Raw and controlled differ by < 0.6 min or 0.4 pp for severe weather.
3. **The delay-rate gap is a lower bound.** Cancelled flights are excluded from the delay rate, yet severe days cancel the most flights (9.0%). Counting cancellations and diversions as disruptions, the severe-vs-clear gap rises from +23.4 to **+27.5 pp** [23.4, 31.3] (+4.1 pp), and moderate from +10.7 to +11.9 pp.
4. **Adjusted for airport, month, hour and airline, severe weather multiplies the odds of a 15+ min delay by 3.6 and of cancellation by 7.7** (both CIs exclude 1 by a wide margin; weather is measured per airport-day, hence the clustered standard errors).
5. **DOT's "weather" cause captures only about a fifth of the excess delay on severe days** (next table).

**DOT weather delay vs excess delay on severe days** (`2c_dot_weather_vs_excess_delay.csv`; severe minus clear, controlled, per flown flight)

| Component | Min/flight | Share of the 25.9-min excess |
|---|---|---|
| DOT "weather" | 5.24 | 20% |
| Air system (NAS) | 5.15 | 20% |
| Late aircraft | 8.53 | 33% |
| Airline | 3.41 | 13% |
| Unattributed (delays < 15 min carry no cause) | 3.56 | 14% |

NAS delay rises by about as much as the weather code, consistent with weather-driven NAS holds (ground stops, flow control) being coded as air system. DOT weather + NAS explain ~40% of the excess. Late-aircraft delay (33%) is secondary propagation, which weather also triggers (Step 3 quantifies propagation). We do **not** claim that 80% of the excess is miscoded weather: only the NAS portion is plausibly miscoding. The cause mix confirms the shift: weather is 16.9% of delay minutes on severe days vs 2.7% on clear days.

### 6.4 Sensitivity: warehouse 4-level label
Rerunning the controlled comparison with the warehouse's own label (`2c_sensitivity_warehouse_label_gaps.csv`) leaves the conclusions unchanged: Severe vs Clear controlled gap +24.8 pp delay rate [21.1, 28.0], +26.7 min, +6.9 pp cancellations — all within the CIs above. The warehouse "Moderate" gap (+11.2 pp) is also close to ours (+10.7 pp). Its "Light" tier (rain < 5 mm) still adds +5.5 pp, so our derived "clear" class contains some mild-weather days; this makes our gaps slightly conservative.

### 6.5 Coverage of the controlled comparison
(`2c_coverage_matched_cells.csv`, `2c_unmatched_*.csv`) The controlled estimate uses 90.7% of severe and 95.0% of moderate flown flights (186,739 of 205,938 severe; 577,007 of 607,533 moderate). All unmatched severe flights (19,199) are in airport-month cells with fewer than 50 severe flights, i.e. small airports with a handful of severe days; no single airport accounts for more than 1.4% of the unmatched severe flights, and months are mildly skewed to winter (Dec 12.9%, Jan 12.4%, Feb 11.3%). Exclusion is therefore diffuse, not concentrated in hubs.

### 6.6 Limitations of the weather analysis
- **One station per airport** (median 0.85 km, up to 49.8 km), same-day observed weather; no hour-level timing, so a morning storm and an evening storm are the same "severe day".
- **Destination and en-route weather are not modeled** here. Flights leaving a clear origin can still be cancelled or delayed by weather elsewhere: 45.5% of cancellations on origin-clear days are weather-coded. This attenuates the gaps (clear days contain weather-affected flights), another reason the estimates are conservative.
- **October excluded**, plus 0.71% of flights at airports with no station (SJU, STT, ECP ...).
- **Association, not proof of causation.** The controlled design holds airport and month fixed but not day-of-week, hour or airline; the logistic model adds hour and airline and agrees. The delay-rate comparison excludes cancelled flights (lower bound, finding 3).
- **DOT cause coding is a reporting convention**, not an engineering measurement; late-aircraft delay mixes original and propagated causes.

## 7. O4 — Delay propagation via aircraft tail_number (Step 3)

**Method.** For every flight with a `tail_number` (5,804,358 flights, 4,897 aircraft), legs are ordered per (aircraft, scheduled date) by scheduled departure (ties: flight number, key) with DuckDB window functions (`LAG`) over *all* scheduled legs, so a cancelled predecessor breaks the chain rather than being skipped. Local clock times are converted to an absolute minute axis (actual arrival aligned to the right day via departure + elapsed time, which handles midnight rollover and time-zone shifts). Two sets:
- **Analysis set** (3,917,656 flights): previous leg same aircraft and day, flown, previous destination = this origin, airports known (so October is out), planned turn >= 0. The actual turnaround may be negative (the aircraft arrived after this flight's scheduled departure) — those are the worst cases and are *kept* for the descriptive analysis.
- **Feature set** (3,693,636 flights, 63.6% of flights with a tail): additionally previous actual arrival <= this scheduled departure, so the value is known at prediction time. Only here are `prev_leg_arr_delay` and `turnaround_min` filled (table `propagation_features`; `results/propagation/propagation_features.parquet`). The 224,020 analysis-set flights whose predecessor arrived after the scheduled departure are *not* in the feature set — they are disproportionately the most delayed turns (see limitations). *(Count note: 224,020 counts every such row in the table, including 6,367 where this flight was itself later cancelled or diverted; the 217,653 quoted in Section 8.1 is the flown subset that has a delay outcome and is the population the leak check used. 224,020 = 217,653 + 6,367.)*

Chain integrity: 98.9% of flown previous legs end at this flight's origin (4,326,737 of 4,375,805), so tail-number ordering reproduces physically consistent aircraft routes. Median planned turn is 46 min.

**Key table — departure delay vs incoming delay** (`3A_dep_delay_vs_incoming_delay.csv`)

| Previous-leg arrival delay | Flights | Mean departure delay (min) | Departing >= 15 late |
|---|---|---|---|
| < -15 (early) | 741,649 | 0.0 | 6.6% |
| 0..14 | 796,657 | 6.9 | 16.5% |
| 15..29 | 271,553 | 19.1 | 54.1% |
| 30..59 | 203,632 | 37.0 | 83.2% |
| 60..89 | 80,824 | 62.6 | 89.1% |
| 120..179 | 36,400 | 116.5 | 89.8% |
| 180+ | 21,179 | 164.8 | 84.4% |

First leg of the day (no inbound aircraft): mean 5.0 min, 10.9% delayed >= 15.

**Findings**
1. **Late aircraft are the dominant source of delay downstream:** 60.8% of departure delays >= 15 min (475,956 of 783,443 in the analysis set) follow an inbound leg that arrived >= 15 min late.
2. **Propagation is nearly one-for-one for short turns:** a 15–29 min late inbound lifts the chance of a >= 15 min late departure from 16.5% (on-time inbound) to 54.1%; 30–59 min late inbound gives 83.2%; the mean departure delay tracks the inbound delay (41.8 -> 37.0, 72.7 -> 62.6 min).
3. **Buffer is the absorber** (`3B_buffer_absorption.csv`). With a planned turn < 45 min nothing is absorbed (a 15–29 min late inbound leaves 24–30 min late); at 60–89 min planned, about half (49%) of a 15–29 min delay and 37% of a 30–59 min delay is absorbed; at 90–119 min, 73–77% of delays under 1 h (58% for 1–2 h); at 120–179 min, 74–84% of delays up to 2 h. Delays of 2 h+ need a 180+ min turn to be mostly absorbed (86%).
4. **Delay accumulates and echoes through the aircraft's day** (`3C_*`). Mean departure delay rises from 5.0 min on leg 1 to 16.4 on leg 7 (arrival delay 0.0 -> 13.4). Aircraft whose first leg arrived 60+ min late (mean 143.8 min) still average 75.6 min late on leg 2 (about half the excess over the on-time baseline), 58.5 on leg 3 and 37.1 on leg 6, vs 9.2 for aircraft that started on time. Aircraft recover about 2–5 min per leg in the air (arrival delay is 2.3–5.2 min lower than departure delay on every leg), consistent with schedule padding.
5. **Absorbing vs amplifying** (`3D_*`; departure delay observed minus expected for the same incoming delay, planned buffer and time of day; 654,873 turns with >= 15 min late inbound). Best absorbers: SkyWest (OO) -8.2 min, Hawaiian (HA) -8.0, Atlantic Southeast (EV) -4.9, Alaska (AS) -4.0. Worst amplifiers: United (UA) +8.8, JetBlue (B6) +5.8, Spirit (NK) +3.1. Southwest is neutral (+0.5) but absorbs only 0.7% of inbound delay in absolute terms because its planned turns are short. Among airports (>= 500 delayed-inbound turns, 121 airports) the best absorbers are small airports (ITO -12.4, GRK -11.2, SBA -10.2) and the worst are PBI (+8.4), JFK (+7.3), MCO (+6.2), BWI (+6.1).

**Validation against DOT `late_aircraft_delay_min`** (delayed arrivals in the analysis set; `3E_*`): DOT flags late-aircraft delay on 86–94% of delayed flights whose inbound was >= 15 min late, but on only 23.9% when our measured inbound delay is < 15 min. Agreement between "our inbound >= 15 min late" and "DOT late-aircraft > 0" is 84.4% (P(DOT | ours) = 91.2%, P(ours | DOT) = 82.3%). The Pearson correlation between our inbound delay and DOT's late-aircraft minutes is 0.83. DOT's late-aircraft share of a flight's delay rises with our inbound delay: 39% (15–29 min) -> 63% (30–59) -> 77% (60–119) -> 83% (120+). 77.3% of all DOT late-aircraft minutes (non-October) sit on flights where we observe a >= 15 min late inbound leg. The first leg of the day carries a late-aircraft flag on 10.5% of its delayed arrivals (overnight carry-over from the previous day, which a same-day chain cannot see). The independent LAG reconstruction therefore reproduces DOT's attribution closely.

**Limitations**
- Chains are same-day only; aircraft delays carried overnight are invisible (the 10.5% first-leg flag). Cancelled or diverted predecessors and non-continuous routes (1.1%) break the chain.
- The leg-number profile mixes position with time of day (later legs fly later); the by-period curves in `3A_dep_delay_vs_incoming_delay_by_period.csv` and `3C_by_leg_and_period.csv` hold hour mix fixed. "Ground change" is reported only for late-inbound turns because early arrivals cannot depart early.
- The absorb/amplify scores standardize for incoming delay, planned buffer and time of day, **not** for hub/airport (airline scores) or airline (airport scores); regional carriers operating for majors and hub congestion remain confounded. Airport CIs are wide for small airports.
- **Feature-set truncation:** the "known at prediction time" rule removes 224,020 flights (5.7% of the chain flights) whose predecessor arrived after this flight's scheduled departure; these are the most extreme delays, so `prev_leg_arr_delay` understates how much propagation exists. An indicator for "predecessor still airborne/late at scheduled departure" could be added as a separate feature (not used by default).
- Tail-number and schedule data are as reported by carriers to DOT; swaps of aircraft not reflected in the tail number are invisible.

## 8. O5 — Predictive mining (Step 4)

### 8.1 Prediction moment: T-60 (definition and rationale)
**Use case.** An operations desk decides one hour before departure which flights to protect (swap aircraft, hold crews, rebook passengers, alert the gate). The model therefore predicts at **T-60: 60 minutes before the flight's scheduled departure**, and may only use information available at that moment. Two scenarios are reported:
- **Day-ahead planning model** — calendar, schedule, route, airline/airport, weather (observed same-day weather as a proxy for a forecast) and historical delay rates; *no propagation features*.
- **T-60 operational model (main)** — day-ahead features plus the **state of the aircraft's previous leg as of T-60**.

**Why not "known at scheduled departure" (T-0).** The Step 3 rule (use `prev_leg_arr_delay` only if the predecessor arrived before the scheduled departure) left the feature NaN exactly for the 217,653 flown flights (224,020 table rows, of which 6,367 were themselves cancelled/diverted and have no delay outcome) where the predecessor was still late. That NaN was itself a leak: those flights arrive 15+ min late **98.8%** of the time, vs 17.3% for flights with no usable predecessor (`results/propagation/` check run before the rebuild). Those columns are now stored as `legacy_*` and are not used for modeling. An explicit "pending at scheduled departure" feature is deliberately **not** included.

**T-60 features** (table `propagation_t60`, built with window functions; every actual predecessor time is placed on an absolute-minute axis in the clock of this flight's origin airport: gate-in = arrival clock aligned to the right day, gate-out = gate-in − elapsed time, so time zones and midnight rollover are handled):

| Feature | Definition at T-60 |
|---|---|
| `prev_leg_status` | `landed` (gate-in <= T-60), `airborne` (left the gate, not yet at the destination gate; includes taxi phases), `not_departed` (gate-out > T-60), `no_prev_leg` (first leg of the aircraft's day, or no usable predecessor: cancelled/diverted, not ending at this origin, unknown airport, no tail number) |
| `prev_leg_arr_delay` | predecessor's arrival delay, only if `landed`, else NaN |
| `prev_leg_dep_delay` | predecessor's departure delay, only if `landed` or `airborne`, else NaN |
| `prev_leg_sched_arr_minus_t60` | minutes from T-60 to the predecessor's *scheduled* arrival (schedule-known) |
| `planned_turn_min` | this scheduled departure − predecessor scheduled arrival (schedule-known; negative in 0.7% of chains where the schedule itself overlaps) |
| `leg_number_of_day` | position in the aircraft's scheduled day |

Status is explicit, so NaN is never the only signal; no flights are dropped (flights without a tail number get `no_prev_leg`). Distribution in the modeling population: airborne 45.5%, landed 27.7%, no predecessor 24.9%, not departed 1.9%. At T-60 the median predecessor is still airborne (median planned turn is 46 min), which is why the predecessor's arrival delay is unknown for 72% of flights while its departure delay is known for 73%. Observed delay rate by status: not departed 95.1%, airborne 23.6%, no predecessor 15.0%, landed 10.5% (landed predecessors are mostly early or on time).

**Leakage controls.** (1) A hard assertion (`ml_common.assert_no_leakage`) rejects any feature that is a forbidden column (`departure_delay_min`, any actual time of this flight, taxi, air time, elapsed time, the five cause columns, cancellation reason, `arrival_delay_min`) or is not on the approved whitelist, and it checks in the database that no predecessor-derived value uses an event later than T-60 (0 violations in 5,804,358 rows; negative tests confirm it fires). (2) `arrival_delay_min` is used only to build the target. (3) Historical delay rates (route, airline, origin) use only **training** months and are expanding-window (a row sees only earlier training months; January has none → NaN; validation/test rows see all eight training months), so no row's own label enters its feature. (4) Weather is same-day observed weather used as a proxy for a forecast.

**Time-zone / midnight spot check** (`results/model_comparison/t60_spot_check.txt`, script `04a_t60_spot_check.py`): three aircraft-days re-derived independently in pandas and asserted equal to the SQL table for every leg — N841VA 2015-02-02 (JFK→SFO then SFO→SAN→SFO; tz shift −3.0 h, gate-out re-expressed in the destination clock), N523JB 2015-05-18 (a SFO→FLL leg with +3.0 h shift; the predecessor lands at 00:19 on 05-19 while the next flight is scheduled for 23:20 on 05-18, correctly `not_departed` at its T-60), N470AA 2015-09-09 (all four statuses in one day, including a −6 min planned turn).

### 8.2 Protocol
- **Population and split** (October excluded): train Jan–Aug **3,836,893** flights (20.2% delayed), validation Sep **462,153** (13.0%), test Nov–Dec **932,084** (17.95%). *Prevalence differs across the three periods, so PR-AUC values are only comparable within one period; every table therefore shows the test base rate and the lift (PR-AUC / base rate).* 43 features; target = arrival delay >= 15 min; flown flights only.
- **Class weights** (identical for every model that accepts weights): w0 = 0.627, w1 = 2.472 (balanced, computed on train). KNN has no weight support (uniform votes); Naive Bayes uses balanced priors.
- **Threshold:** max-F1 on the September validation set, applied unchanged to test (F1, precision, recall, accuracy). PR-AUC and ROC-AUC are threshold-free. Accuracy is not shown as a ranking metric: the majority dummy scores 82% accuracy with zero recall.
- **Seeds:** 3 seeds (42, 43, 44) for LightGBM, XGBoost (default and tuned, ordinal), Random Forest, KNN, Decision Tree; deterministic models (Logistic Regression, Naive Bayes, native-categorical XGBoost variants) are single runs. Mean ± std over seeds is reported.
- **Tuning (validation only, never the test set).**
  - *LightGBM:* 16 configurations (num_leaves {63, 255} x learning_rate {0.1, 0.05} x min_child_samples {100, 1000} x feature_fraction {0.6, 0.9}) on a 1M-row stratified train sample, early stopping (50 rounds) on the September set with metric average precision. Best: num_leaves 63, learning rate 0.05, min_child_samples 100, feature_fraction 0.6; final models refit on all training rows with early stopping (417–556 rounds).
  - *XGBoost (same budget, same rows, same validation set, same early stopping):* 16 configurations (max_depth {6, 8} x learning_rate {0.1, 0.05} x min_child_weight {5, 50} x colsample_bytree {0.6, 0.9}), identical 1M stratified sample and September validation set, early stopping 50 rounds on `aucpr` (= PR-AUC), selected by validation PR-AUC. Best: max_depth 8, learning rate 0.05, min_child_weight 50, colsample 0.6. Both use balanced sample weights and subsample/bagging 0.8. The "default params" XGBoost rows (depth 8, eta 0.1, min_child_weight 5, colsample 0.8) are kept to show the effect of tuning. **Categorical handling:** the main XGBoost rows use *ordinal integer codes* (`enable_categorical=False`); separate rows use XGBoost's *native categorical* support (`enable_categorical=True`, `max_cat_to_onehot=4`). Numerics keep native NaN in both. LightGBM uses native categoricals and NaN.
  - *Decision Tree:* depth tuned on validation over {3,...,20} (best 12, 1M sample); Random Forest 150 trees, min_samples_leaf 50 (not tuned); Logistic Regression C = 1 (not tuned).
- **Sample flags.** Random Forest is trained on a **1M-row stratified sample** (pre-emptive choice to keep runtime in minutes; its measured 38 s at 1M suggests the full set may have been feasible, which is why matched-data rows are reported). KNN uses a **200k stratified sample** (k = 50, numeric features only) and is evaluated on the full test set. Sklearn models use median imputation + missing indicators fitted on train only; Logistic Regression uses one-hot (min frequency 200) + scaling; tree/NB models use ordinal codes.

### 8.3 Model comparison (T-60 operational feature set, test set)
### (a) T-60 test set, one row per model (test base rate 0.1795; threshold = max-F1 on September)
| Model | Train rows | PR-AUC (mean ± std over seeds) | Lift | ROC-AUC | F1 | Precision | Recall | Train s | Predict s | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost (native categoricals, tuned) | 3,836,893 | 0.6409 | 3.57x | 0.8214 | 0.561 | 0.600 | 0.527 | 196 | 6.6 | 1 |
| XGBoost (tuned, ordinal codes) | 3,836,893 | 0.6390 ± 0.0006 | 3.56x | 0.8191 ± 0.0005 | 0.560 | 0.594 | 0.530 | 84 | 1.7 | 3 |
| LightGBM (main) | 3,836,893 | 0.6387 ± 0.0001 | 3.56x | 0.8202 ± 0.0000 | 0.560 | 0.584 | 0.538 | 134 | 3.5† | 3 |
| XGBoost (native categoricals, default params) | 3,836,893 | 0.6364 | 3.54x | 0.8173 | 0.558 | 0.592 | 0.528 | 51 | 1.3 | 1 |
| XGBoost (default params, ordinal codes) | 3,836,893 | 0.6350 ± 0.0012 | 3.54x | 0.8165 ± 0.0010 | 0.556 | 0.597 | 0.521 | 115 | 0.8 | 3 |
| Random Forest | 1,000,000 | 0.6189 ± 0.0006 | 3.45x | 0.8128 ± 0.0001 | 0.550 | 0.593 | 0.512 | 36 | 2.0 | 3 |
| Decision Tree (CART) | 3,836,893 | 0.6041 ± 0.0000 | 3.36x | 0.7929 ± 0.0000 | 0.533 | 0.647 | 0.453 | 75 | 0.2 | 3 |
| Logistic Regression | 3,836,893 | 0.5502 | 3.06x | 0.7806 | 0.507 | 0.521 | 0.494 | 352 | 1.5 | 1 |
| KNN | 200,000 | 0.4784 ± 0.0029 | 2.66x | 0.7400 ± 0.0023 | 0.460 | 0.512 | 0.417 | 0 | 591.7 | 3 |
| Naive Bayes (Categorical) | 3,836,893 | 0.4593 | 2.56x | 0.7318 | 0.429 | 0.512 | 0.369 | 105 | 0.8 | 1 |
| Naive Bayes (Gaussian) | 3,836,893 | 0.3624 | 2.02x | 0.7198 | 0.436 | 0.340 | 0.607 | 7 | 0.6 | 1 |
| Majority dummy | 3,836,893 | 0.1795 | 1.00x | 0.5000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1 |
† LightGBM model-only prediction time on 932,084 rows (3 runs); the cached training log also timed pandas slicing (62 s). Random Forest uses a 1M stratified sample, KNN a 200k stratified sample.

*Findings*
1. **Gradient-boosted trees are clearly the best family:** LightGBM, tuned XGBoost (ordinal or native) and default XGBoost reach PR-AUC 0.635–0.641 (lift 3.5–3.6x, ROC-AUC 0.817–0.821); Random Forest 0.619, a single CART 0.604, Logistic Regression 0.550, KNN 0.478, Naive Bayes 0.459 / 0.362. The majority dummy is at the base rate (0.1795).
2. **LightGBM is not the unique winner.** At T-60 it is statistically tied with tuned XGBoost on ordinal codes (difference +0.0004, Section 8.4), and XGBoost with native categoricals is nominally higher (0.6409 single run, +0.0021 over LightGBM; practically negligible).
3. **At F1 ~ 0.56 the boosted models predict about 16–17% of flights as delayed** (precision 0.58–0.60, recall 0.52–0.54) against a true rate of 18%.
4. **Runtime:** LightGBM trains in 134 s (417–556 early-stopped rounds), tuned XGBoost (ordinal) in 84 s, native-categorical XGBoost in 196 s; prediction of 932k flights takes 3.5 s (LightGBM, model-only†) and 1.7 s (XGBoost). KNN is slow at prediction time (592 s ± 361 for the 932k test queries; the spread reflects machine load, not the method) and Logistic Regression slow to fit (352 s).

### 8.4 Matched training data and paired statistical tests
### (b) Matched training data
| training_data | model | PR-AUC / diff | Lift / verdict | PR-AUC 95% CI (flight-level) | roc_auc | train_s |
|---|---|---|---|---|---|---|
| 1M stratified sample | LightGBM on RF's 1M sample | 0.6333 ± 0.0004 | 3.53x | [0.6310, 0.6356] | 0.8158 | 16.1 |
| 1M stratified sample | XGBoost (tuned) on RF's 1M sample | 0.6376 ± 0.0006 | 3.55x | [0.6351, 0.6396] | 0.8185 | 38.0 |
| 1M stratified sample | Random Forest | 0.6189 ± 0.0006 | 3.45x | [0.6160, 0.6208] | 0.8128 | 35.6 |
| 1M stratified sample |   >> LightGBM minus Random Forest (paired) | +0.0149 | statistically real and practically meaningful | [0.0141, 0.0157] |  |  |
| 200k stratified sample | LightGBM on KNN's 200k sample | 0.6223 ± 0.0006 | 3.47x | [0.6200, 0.6246] | 0.8081 | 22.1 |
| 200k stratified sample | XGBoost (tuned) on KNN's 200k sample | 0.6299 ± 0.0007 | 3.51x | [0.6271, 0.6317] | 0.8133 | 27.3 |
| 200k stratified sample | KNN | 0.4784 ± 0.0029 | 2.66x | [0.4775, 0.4826] | 0.74 | 0.3 |
| 200k stratified sample |   >> LightGBM minus KNN (paired) | +0.1422 | statistically real and practically meaningful | [0.1405, 0.1439] |  |  |
| full training set (3.84M rows) | LightGBM (main) | 0.6387 ± 0.0001 | 3.56x | [0.6366, 0.6411] | 0.8202 | 134.2 |

### (c) LightGBM minus XGBoost, paired PR-AUC difference
| Comparison (T-60) | Mean diff | 95% CI, flight-level | 95% CI, day-clustered | Verdict (flight-level CI) |
|---|---|---|---|---|
| LightGBM minus XGBoost (tuned, ordinal codes) | +0.0004 | [-0.0002, +0.0011] | [-0.0019, +0.0031] | not statistically distinguishable (practically equivalent, CI inside +/-0.01) |
| LightGBM minus XGBoost (default params, ordinal codes) | +0.0031 | [+0.0025, +0.0037] | [+0.0007, +0.0053] | statistically real but practically negligible (< 0.01) |
| LightGBM minus XGBoost (native categoricals, tuned) | -0.0021 | [-0.0025, -0.0017] | [-0.0035, -0.0009] | statistically real but practically negligible (< 0.01) |
| LightGBM minus XGBoost (native categoricals, default params) | +0.0024 | [+0.0020, +0.0028] |  | statistically real but practically negligible (< 0.01) |

| Comparison (day-ahead) | Mean diff | 95% CI, flight-level | 95% CI, day-clustered |
|---|---|---|---|
| LightGBM minus XGBoost (tuned, ordinal codes) | +0.0111 | [+0.0101, +0.0121] | [+0.0043, +0.0181] |
| LightGBM minus XGBoost (default params, ordinal codes) | +0.0273 | [+0.0261, +0.0285] |  |
| LightGBM minus XGBoost (native categoricals, tuned) | -0.0037 | [-0.0045, -0.0029] | [-0.0083, +0.0011] |

*Method note.* Paired differences use the same bootstrap weights for both models. The flight-level interval treats the 932,084 test flights as independent; flights on the same day share weather, congestion and aircraft, so we also report a **day-clustered** bootstrap (61 test days resampled, 1,000 resamples), which is 3–4x wider and is the more honest uncertainty. "Practically meaningful" means a difference >= 0.01 PR-AUC.

*Findings*
1. **LightGBM minus tuned XGBoost (ordinal codes), T-60: +0.0004, flight-level 95% CI [-0.0002, +0.0011], day-clustered [-0.0019, +0.0031]. Not statistically distinguishable and practically equivalent (CI well inside +/-0.01).** Against XGBoost with native categoricals, LightGBM is 0.0021 lower (flight-level CI [-0.0025, -0.0017]; day-clustered [-0.0035, -0.0009]): statistically detectable, practically negligible.
2. **The algorithm effect is small; the family effect is large.** On identical rows, LightGBM beats Random Forest by +0.0149 [0.0141, 0.0157] on the 1M sample and KNN by +0.1422 [0.1405, 0.1439] on the 200k sample. KNN's deficit is an algorithm effect, not a training-size effect (a boosted model on the same 200k rows reaches 0.622–0.630).
3. **On identical samples tuned XGBoost is ahead of LightGBM** (0.6376 vs 0.6333 on 1M; 0.6299 vs 0.6223 on 200k), a possible reason is that LightGBM's hyperparameters were not re-tuned for the smaller samples (not tested). This counts *against* a claim that LightGBM is more data-efficient.
4. **Day-ahead set (no propagation features):** LightGBM leads tuned XGBoost with ordinal codes by +0.0111 (flight-level [0.0101, 0.0121]; day-clustered [0.0043, 0.0181]) and Random Forest by +0.0107, but **XGBoost with native categoricals is at least as good (LightGBM minus it: -0.0037, day-clustered [-0.0083, +0.0011])**. LightGBM's day-ahead lead therefore comes from native handling of high-cardinality categoricals (route has 8.6k levels), which XGBoost 2.x also offers, not from the boosting algorithm.

### 8.5 Ablation: what each objective adds (LightGBM)
### (d) LightGBM ablation (test; CIs flight-level bootstrap on seed 42)
| setting | n_features | seeds | pr_auc | pr_auc_ci | roc_auc | roc_auc_ci |
|---|---|---|---|---|---|---|
| Day-ahead, weather OFF | 21 | 1 | 0.2479 | [0.2463, 0.2494] | 0.6200 | [0.6186, 0.6214] |
| Day-ahead, weather ON | 37 | 3 | 0.3539 ± 0.0014 | [0.3500, 0.3544] | 0.6990 ± 0.0003 | [0.6973, 0.6999] |
| Day-ahead + schedule-known turn structure (weather ON) | 40 | 1 | 0.4174 | [0.4152, 0.4197] | 0.7242 | [0.7229, 0.7256] |
| T-60, weather OFF | 27 | 1 | 0.5983 | [0.5960, 0.6007] | 0.7805 | [0.7791, 0.7819] |
| T-60, weather ON  [main] | 43 | 3 | 0.6387 ± 0.0001 | [0.6366, 0.6411] | 0.8202 ± 0.0000 | [0.8190, 0.8214] |
| T-60 without month / week_of_year | 41 | 1 | 0.6433 | [0.6411, 0.6455] | 0.8233 | [0.8221, 0.8246] |
| T-60 without historical delay rates | 40 | 1 | 0.6386 | [0.6365, 0.6410] | 0.8203 | [0.8192, 0.8216] |

Paired effects:
| effect | delta_pr_auc | pr_auc_ci | delta_roc_auc | roc_auc_ci |
|---|---|---|---|---|
| Weather effect, day-ahead (ON minus OFF) | 0.1043 | [0.1023, 0.1062] | 0.0785 | [0.0772, 0.0798] |
| Weather effect, T-60 (ON minus OFF) | 0.0405 | [0.0395, 0.0414] | 0.0396 | [0.0388, 0.0405] |
| Propagation effect: T-60 minus day-ahead (weather ON) | 0.2866 | [0.2844, 0.2889] | 0.1216 | [0.1204, 0.1228] |
|   of which schedule-known turn structure | 0.0652 | [0.0637, 0.0667] | 0.0257 | [0.0250, 0.0263] |
|   of which live predecessor status / delays | 0.2214 | [0.2194, 0.2234] | 0.0959 | [0.0948, 0.0969] |
| Month / week_of_year effect (with minus without) | -0.0045 | [-0.0049, -0.0040] | -0.0031 | [-0.0035, -0.0028] |
| Historical-rate effect (with minus without) | 0.0001 | [-0.0001, 0.0004] | -0.0002 | [-0.0004, 0.0000] |

*Findings*
1. **Delay propagation (O4) is the most valuable signal:** moving from day-ahead to T-60 raises PR-AUC from 0.354 to 0.639 (+0.287, CI [0.284, 0.289]; lift 1.97x -> 3.56x; ROC-AUC 0.699 -> 0.820). Of that, +0.065 comes from schedule-known turn structure (planned turn, leg number, predecessor's scheduled arrival relative to T-60) and **+0.221 from the live status of the previous leg** (status, departure delay, arrival delay).
2. **Weather (O3) adds +0.104 PR-AUC in the day-ahead model** (0.248 -> 0.354) **but only +0.041 at T-60** (0.598 -> 0.639): once the aircraft's own delay state is known, much of what weather would explain is already visible. Weather is still a statistically and practically meaningful addition in both settings.
3. **Month / week_of_year hurt out of sample:** removing them *raises* test PR-AUC by +0.0045 (0.6433 vs 0.6387), consistent with trees being unable to extrapolate calendar position from Jan–Aug training to Nov–Dec. Historical delay rates add nothing at T-60 (+0.0001, CI includes 0), because the predecessor status and airline/airport/route features already carry that information. (Both are reported as findings; neither was removed from the main model, which was fixed before the ablation.)
4. The day-ahead model is weak (lift 1.97x): without live information, delay is hard to predict a day ahead from schedule, route and weather alone.

### 8.6 Scalability
| Training rows | LightGBM s (val PR-AUC) | XGBoost s (val PR-AUC) | Logistic Regression s (val PR-AUC) | Random Forest s (val PR-AUC) |
|---|---|---|---|---|
| 100,000 | 4.2 (0.5332) | 1.5 (0.5316) | 1.4 (0.4625) | 2.7 (0.5238) |
| 500,000 | 7.9 (0.5522) | 4.0 (0.5428) | 7.5 (0.4648) | 17.2 (0.5391) |
| 1,000,000 | 11.3 (0.5574) | 7.0 (0.5529) | 14.9 (0.4648) | 38.4 (0.5428) |
| 3,836,893 | 28.8 (0.5640) | 28.4 (0.5592) | 61.6 (0.4667) | not run |

(300 boosting rounds, tuned parameters, no early stopping; Logistic Regression includes one-hot preprocessing; Random Forest was not run on the full training set.) Training time grows sub-linearly for LightGBM (28.8 s at 3.8M rows vs 4.2 s at 100k: 6.9x for 38x the data) and faster for XGBoost (28.4 s vs 1.5 s: 19x); at the full size the two are equal, and at small sizes XGBoost is faster. Validation PR-AUC still rises between 1M and the full set (LightGBM 0.557 -> 0.564), so the boosted models benefit from more data, while Logistic Regression has saturated (0.4625 -> 0.4667). Random Forest's 38 s at 1M rows scales roughly linearly.

### 8.7 Explainability
**Importance.** Gain share by feature group (LightGBM, `results/explain/lgbm_gain_by_group.csv`): predecessor status/delay 42.3%, airline/airport/route 24.3%, weather (origin + destination) 11.7%, schedule turn structure 11.5%, time of day/calendar 9.1%, historical rates 1.0%, distance/schedule 0.2%. Top-10 SHAP features are in table (e) below; `prev_leg_dep_delay` is the largest driver (mean |SHAP| 0.51 log-odds). `week_of_year` is the #2 SHAP feature (0.24) yet removing it improves test PR-AUC (Section 8.5): the model uses it heavily within the training months, and the pattern does not transfer to Nov–Dec. Plots: `shap_summary.png`, `shap_dependence_*.png`, `lgbm_gain_importance.png`.

### (e) Top 10 SHAP features (mean |SHAP|, log-odds, 50k test flights)
| feature | mean_abs_shap |
|---|---|
| prev_leg_dep_delay | 0.505 |
| week_of_year | 0.245 |
| d_prcp | 0.173 |
| sched_dep_hour | 0.168 |
| route | 0.167 |
| prev_leg_status | 0.147 |
| origin | 0.139 |
| o_prcp | 0.138 |
| prev_leg_arr_delay | 0.123 |
| prev_leg_sched_arr_minus_t60 | 0.111 |

Gain by feature group:
| feature_group | gain_share_pct |
|---|---|
| Predecessor status/delay (T-60) | 42.27 |
| Airline / airport / route | 24.28 |
| Weather (origin+dest) | 11.69 |
| Schedule turn structure | 11.49 |
| Time of day / calendar | 9.11 |
| Historical delay rates | 1.01 |
| Distance / schedule | 0.15 |

**Interaction check (snow x northern airport x winter).** Counterfactual: raise origin snowfall from 0 to 30 mm (and snow depth +100 mm) and measure the change in predicted delay probability. Northern origins: +6.5 percentage points; southern origins: +5.3 pp (December test flights). The effect is the same in November and December. The model therefore learned a modestly larger snow effect for northern airports, but there is **no evidence for a winter-specific interaction**: the training months contain no Nov/Dec, so calendar splits cannot carry a winter pattern forward. We do not claim the three-way interaction.

**Native handling of missing weather and rare categories (subgroup PR-AUC, test).**
| Subgroup (test) | Flights | Delayed rate | LightGBM | XGBoost tuned (ordinal) | XGBoost tuned (native cat.) | Random Forest | Logistic Regression |
|---|---|---|---|---|---|---|---|
| All test flights | 932,084 | 0.18 | 0.6388 | 0.6384 | 0.6409 | 0.6183 | 0.5502 |
| Origin weather missing (SJU, STT, ... NaN weather) | 6,927 | 0.151 | 0.5871 | 0.5895 | 0.5887 | 0.5613 | 0.5352 |
| Origin snow > 0 | 30,259 | 0.405 | 0.7475 | 0.7526 | 0.7482 | 0.728 | 0.6914 |
| Rare routes (<100 training flights) | 10,822 | 0.193 | 0.6288 | 0.6261 | 0.6277 | 0.6123 | 0.5619 |
| Predecessor 'not_departed' at T-60 | 15,679 | 0.954 | 0.9913 | 0.9928 | 0.9919 | 0.9754 | 0.9792 |
| Predecessor 'landed' at T-60 | 280,440 | 0.094 | 0.2359 | 0.2307 | 0.2402 | 0.2209 | 0.1769 |
| No predecessor (first leg / broken chain) | 235,496 | 0.149 | 0.3796 | 0.3818 | 0.3845 | 0.3663 | 0.31 |

On the 6,927 flights from airports without a weather station, both boosted models (0.587–0.590) are well above Logistic Regression with median imputation (0.535) and Random Forest (0.561); LightGBM is not better than tuned XGBoost (0.5871 vs 0.5895), so native NaN handling is a boosted-tree property, not a LightGBM-specific one. On rare routes (<100 training flights) LightGBM is marginally higher than XGBoost (0.6288 vs 0.6261–0.6277), within noise.

**Decision Tree rules (readable features, depth 4; `decision_tree_depth4_rules.txt`, depth 5 also saved).** The five most useful rules, with support and observed delay rate on train and on the Nov–Dec test set (rates are delay frequencies in the leaf, not the model's balanced vote):
1. IF the predecessor has **not left the gate at T-60** (not Hawaiian, distance > 165 miles) THEN delayed — **98%** train / 97% test (n = 65,417 / 13,684).
2. IF the predecessor is **en route, left the gate > 31.5 min late, and the planned turn is <= 63.5 min** THEN delayed — **85%** / 83% (n = 140,186 / 30,470).
3. IF the predecessor is en route, left 16.5–31.5 min late, and the planned turn is <= 43.5 min THEN delayed — 51% / 51% (n = 77,590 / 16,818).
4. IF the predecessor is due in less than ~10 min before this departure (scheduled arrival minus T-60 > 49.5 min) and has already left THEN delayed — 88% / 87% (n = 25,835 / 5,425).
5. IF there is no predecessor problem (not "not departed", due at least ~24 min before departure; this includes flights with no predecessor) **and origin weather is clear** THEN on time — only **12%** delayed train / 10% test (n = 2,822,334 / 676,037); the same without clear weather: 23% / 20% delayed (n = 517,149 / 146,104).

The leaf conditions were simplified (redundant bounds merged; thresholds on imputed values removed). **Interpretability trade-off:** the readable depth-4 / depth-5 trees reach PR-AUC 0.4915 / 0.5291 (F1 0.500 / 0.517) against LightGBM's 0.6388, a loss of 0.11–0.15 PR-AUC for the readability; an unrestricted-depth CART on all features (depth 12) reaches 0.6041.

### 8.8 Threshold-transfer diagnostic (F1 / recall caveat)
### (f) Threshold transfer (val prevalence 13.0% -> test prevalence 17.95%)
| model | Sep-tuned threshold | Val F1 (Sep, at its best thr) | Test F1 @ Sep threshold | Best possible test F1 (diagnostic) | Test predicted-positive rate | F1 lost to transfer |
|---|---|---|---|---|---|---|
| XGBoost (native categoricals, tuned) | 0.5554 | 0.5093 | 0.5615 | 0.5616 | 0.1578 | 0.0002 |
| XGBoost (tuned, ordinal codes) | 0.5256 | 0.5071 | 0.5603 | 0.5605 | 0.1604 | 0.0002 |
| LightGBM (main) | 0.5630 | 0.5085 | 0.5602 | 0.5605 | 0.1656 | 0.0002 |
| XGBoost (native categoricals, default params) | 0.5808 | 0.5063 | 0.5583 | 0.5584 | 0.1600 | 0.0001 |
| XGBoost (default params, ordinal codes) | 0.5480 | 0.5015 | 0.5563 | 0.5565 | 0.1568 | 0.0002 |
| Random Forest | 0.5726 | 0.4953 | 0.5495 | 0.5500 | 0.1552 | 0.0004 |
| Decision Tree (CART) | 0.6636 | 0.4761 | 0.5327 | 0.5328 | 0.1255 | 0.0001 |
| Logistic Regression | 0.5506 | 0.4434 | 0.5071 | 0.5075 | 0.1702 | 0.0004 |
| KNN | 0.2600 | 0.4230 | 0.4598 | 0.4673 | 0.1463 | 0.0075 |
| Naive Bayes (Categorical) | 0.9372 | 0.4041 | 0.4287 | 0.4498 | 0.1294 | 0.0210 |
| Naive Bayes (Gaussian) | 0.0583 | 0.3671 | 0.4356 | 0.4437 | 0.3206 | 0.0081 |

F1 is computed at the threshold that maximizes F1 on September, where only 13.0% of flights were delayed, and applied to test (17.95%). Validation and test F1 are not comparable (F1 depends on prevalence), but the transfer itself costs very little for the boosted models: F1 at the September threshold is 0.5602 vs 0.5605 for the best possible test threshold (a loss of 0.0002–0.0004 for LightGBM, XGBoost, Random Forest, Decision Tree and Logistic Regression). The loss is larger for the weaker, less calibrated models (KNN 0.0075, Gaussian NB 0.0081, Categorical NB 0.0210). The "best possible" F1 is a diagnostic computed on the test set and was **not** used to choose anything. Caveat: precision and recall at the September threshold depend on the prevalence shift; at the same threshold LightGBM flags 16.6% of test flights against a true 18.0%.

### 8.9 Why LightGBM? What the evidence supports
The comparison does **not** show that LightGBM is the single best algorithm. It supports these statements:
1. **Accuracy:** gradient boosting is clearly the best family (T-60 PR-AUC 0.639 vs 0.619 Random Forest, 0.604 CART, 0.550 Logistic Regression, 0.478 KNN, 0.459/0.362 Naive Bayes; lift 3.56x). LightGBM is in the top group, **tied with tuned XGBoost** (difference +0.0004 [-0.0019, +0.0031] day-clustered) and 0.0021 below the nominally best native-categorical XGBoost run (practically negligible).
2. **Statistical reality:** every gap to the non-boosted models is real and meaningful even under the day-clustered bootstrap (Random Forest -0.0206 [0.0174, 0.0240]; Logistic Regression -0.0887); gaps among the boosted variants are all below 0.01.
3. **Stability:** across three seeds LightGBM's PR-AUC varies by ±0.0001, XGBoost's by ±0.0006–0.0012, so the reported number is reproducible.
4. **Training/prediction time:** not a LightGBM advantage here. Full-data training takes 134 s (early stopping), longer than ordinal XGBoost (84 s); at a fixed 300 rounds the two are equal on 3.8M rows (28.8 vs 28.4 s) and XGBoost is faster on small samples. LightGBM's cost grows more slowly with data (6.9x for 38x the rows vs 19x). Prediction takes 3.5 s for 932k flights (XGBoost 1.7 s); the 62 s in the raw log is a pandas-slicing artifact.
5. **High-cardinality categoricals:** with route (8.6k levels), origin and destination as features, native categorical handling is worth about +0.011 PR-AUC over ordinal coding on the day-ahead set (LightGBM vs ordinal XGBoost), a real effect. XGBoost with its native categorical mode closes the gap (LightGBM -0.0037 [-0.0083, +0.0011]), so this is a property of native categorical support, not of LightGBM alone; native XGBoost trained 2.3x slower at T-60 (196 s vs 84 s).
6. **Missing weather:** both boosted models handle NaN natively and beat imputation-based models on the weather-missing subgroup (0.587–0.590 vs 0.535–0.561); LightGBM is not better than XGBoost there.
7. **Nonlinear interactions:** boosted trees capture strong threshold interactions (predecessor delay x planned turn x status); the snow x north effect is only modest (+1.2 pp difference) and the winter dimension is untested (no Nov/Dec in training). We do not credit LightGBM with learning a snow x north x winter interaction.
8. **Explainability trade-off:** a depth-5 Decision Tree gives auditable IF-THEN rules (e.g. "predecessor not departed at T-60 -> 98% delayed") but loses 0.11 PR-AUC; LightGBM plus exact TreeSHAP (`pred_contrib`, no sampling error) recovers the same story (predecessor delay and status are the top drivers) at full accuracy. XGBoost would offer the same TreeSHAP; this point does not separate the two.

**Conclusion.** LightGBM is a justified main model because it is as accurate as any model tested, the most seed-stable, scales sub-linearly with data and handles categoricals and NaN natively without encoding choices. It was *not* the unique winner on accuracy: tuned XGBoost ties it (and with native categoricals is nominally ahead by 0.002), so the defensible headline is "gradient-boosted trees win; LightGBM and XGBoost are equivalent within 0.002 PR-AUC".

### 8.10 Limitations of the predictive study
- **Single test period.** Nov–Dec only; prevalence changes across periods (20.2% / 13.0% / 17.95%) and calendar features do not extrapolate (Section 8.5). Results for other seasons are untested.
- **Weather is observed same-day weather**, a proxy for a forecast, so the weather effect is an upper bound on what a real forecast would provide; destination and en-route weather are only partly captured.
- **Day-level dependence.** Flight-level bootstrap CIs are optimistic; day-clustered CIs are wider and are the ones to rely on for small differences.
- **Sample-based baselines.** Random Forest (1M rows) and KNN (200k rows) were not trained on the full set; matched-data rows quantify the effect. LightGBM hyperparameters were not re-tuned for the matched samples; Random Forest and Logistic Regression were not tuned.
- **T-60 information.** 72% of flights have an unknown predecessor arrival delay at T-60 (the predecessor is still airborne); the model relies on the predecessor's departure delay and status.
- Hyperparameters were tuned on a 1M-row sample and refit on the full training set; seeds vary model randomness only, not the train/val/test split.

## 9. Extensions to Step 4: neural network, ensemble, particle-swarm tuning, weather impact and base-paper comparison

All numbers below are on the same Nov–Dec test set (932,084 flights, base rate 0.1795) and the same train (Jan–Aug) / validation (Sep) protocol as Section 8. Scripts: `mining/07_neural_net.py`, `mining/08_post_departure.py`, `mining/09_pso_tuning_final.py`, `mining/pso.py`. Results: `results/model_comparison/nn_and_ensemble_*.csv`, `post_departure_scenario.csv`, `pso_best_params.json`, `mlp_pso_best.json`, `tuned_final_weather_on_off.csv`, `tuned_weather_effect.csv`.

### 9.1 Neural network and ensemble
Embedding MLP (PyTorch; embeddings for the 11 categorical columns, standardised numerics with median imputation and missing indicators, 512-256-128 ReLU layers with batch norm and dropout, class-weighted BCE, AdamW with OneCycle, early stopping on validation PR-AUC, one seed). The ensemble is the mean of rank-normalised scores of LightGBM, tuned XGBoost (native categoricals) and the MLP.

| Task | Model | PR-AUC | ROC-AUC |
|---|---|---|---|
| T-60 | Ensemble (LightGBM + XGBoost + MLP) | **0.6446** | **0.8242** |
| T-60 | XGBoost (native, tuned) | 0.6409 | 0.8214 |
| T-60 | LightGBM | 0.6388 | 0.8201 |
| T-60 | MLP | 0.6338 | 0.8173 |
| Day-ahead | Ensemble | **0.3597** | **0.7028** |
| Day-ahead | XGBoost (native, tuned) | 0.3559 | 0.6996 |
| Day-ahead | LightGBM | 0.3522 | 0.6986 |
| Day-ahead | MLP | 0.3373 | 0.6887 |

The MLP trails the boosted trees (typical for tabular data) but beats Random Forest at T-60 (0.619). The ensemble's lead is about +0.004 PR-AUC; a paired bootstrap on it has not been run, so it should be described as best by a small margin.

### 9.2 Particle-swarm hyperparameter tuning
Particle swarm optimisation (the tuner used by the base paper; 8 particles, 5 iterations, search spaces in `09_pso_tuning_final.py`), maximising validation PR-AUC on a 600,000-row stratified training sample, then refit on all training rows.

| Task | Model | Tuned values | Test PR-AUC | Before PSO |
|---|---|---|---|---|
| T-60 | XGBoost | max_depth 9, lr 0.055, min_child_weight 40, colsample 0.48, gamma 0.11, subsample 0.99 | 0.6407 | 0.6390 |
| T-60 | LightGBM | num_leaves 164, lr 0.05, min_child_samples 674, feature_fraction 0.40, lambda_l2 0.01, bagging 1.0 | **0.6424** | 0.6387 |
| Day-ahead | XGBoost | max_depth 9, lr 0.05, min_child_weight 56, colsample 0.40, gamma 1.02, subsample 1.0 | 0.3437 | 0.3411 |
| Day-ahead | LightGBM | num_leaves 78, lr 0.098, min_child_samples 2000, feature_fraction 0.77, lambda_l2 0.81, bagging 1.0 | 0.3469 | 0.3539 |
| T-60 | MLP | width 676, dropout 0.10, lr 0.0076, weight decay 8.8e-5, batch 1024 | 0.6254 | 0.6338 |
| Day-ahead | MLP | width 781, dropout 0.20, lr 0.0038, weight decay 2.1e-4, batch 1024 | 0.3327 | 0.3373 |

Tuning gives at most +0.004 PR-AUC (T-60 LightGBM) and hurt day-ahead LightGBM and both MLPs: settings chosen on a 600k sample with few epochs did not transfer to full-data training. The Step 4 configurations remain the reference; PSO is reported for completeness and to match the base paper's method.

### 9.3 Weather impact with the tuned models
"Delayed" = arrival delay >= 15 min. Weather features (14 numeric + 2 class columns) switched ON vs OFF, tuned models refit on the full training set, 95% CIs from a paired Poisson bootstrap (300 draws).

| Task | Model | PR-AUC off | PR-AUC on | Gain (95% CI) | ROC-AUC gain |
|---|---|---|---|---|---|
| T-60 | XGBoost | 0.599 | 0.641 | +0.042 [0.041, 0.043] | +0.040 |
| T-60 | LightGBM | 0.599 | 0.642 | +0.043 [0.043, 0.044] | +0.041 |
| Day-ahead | XGBoost | 0.247 | 0.344 | +0.097 [0.095, 0.099] | +0.072 |
| Day-ahead | LightGBM | 0.244 | 0.347 | +0.103 [0.101, 0.105] | +0.077 |

Weather matters most when nothing is known about the aircraft's earlier legs (day-ahead): it roughly doubles the lift over the base rate relative to the weather-free model. At T-60 the predecessor's status already carries most of the signal, so weather adds less, but the gain is still clearly non-zero. Weather accounts for 11.7% of LightGBM's gain (Section 8.7). Caveat (Section 3): weather is observed same-day weather used as a proxy for a forecast, so these effects are upper bounds for a deployed system.

### 9.4 Post-departure scenario (comparison only)
Adds the flight's own departure delay to the T-60 features, matching the base paper's feature set (1)+(3). Departure delay is known only after take-off, so this is not an operational model; the script deliberately bypasses the T-60 leakage whitelist and nothing else imports it.

| Model | PR-AUC | ROC-AUC | Accuracy | Recall | F2 | Accuracy / recall / F2 at ~86% recall |
|---|---|---|---|---|---|---|
| XGBoost (tuned) | 0.880 | 0.944 | 93.1% | 75.8% | 77.3% | 87.9 / 86.3 / 79.9 |
| LightGBM | 0.879 | 0.943 | 93.1% | 75.9% | 77.3% | 88.4 / 85.8 / 80.0 |

### 9.5 Comparison with the base paper
Base paper: Schösser & Schönberger (2022), *On the Performance of Machine Learning Based Flight Delay Prediction*, Promet – Traffic&Transportation 34(6), 825–838. Same 2015 DOT data and >= 15 min rule. Its three feature sets map onto ours: set (1) = day-ahead, (1)+(2) (previous-flight delay) = T-60, (1)+(3) (departure delay) = post-departure.

Differences that limit direct comparison: the paper uses the 30 busiest airports (2.23M flights), a random 75:25 split, undersampled training data (pushing recall up and accuracy down) and reports accuracy, recall and F2 from single runs. We use all airports, a time-based split, natural class balance with a validation-tuned threshold, and report PR-AUC/ROC-AUC with confidence intervals.

| Setup | Paper XGBoost (acc / recall / F2, %) | Ours at a similar recall (acc / recall / F2, %) |
|---|---|---|
| Day-ahead | 69.5 / 68.2 / 57.2 | ensemble 62.9 / 67.3 / 52.5 |
| T-60 | 70.9 / 68.5 / 58.1 | ensemble 79.2 / 67.1 / 61.0 |
| Post-departure | 89.9 / 83.4 / 80.3 | XGBoost 87.9 / 86.3 / 79.9 |

Conclusions: the model ranking agrees with the paper (tree ensembles above the neural network and simpler baselines); the previous-leg delay helps in both studies, much more in ours (PR-AUC 0.36 -> 0.64); at T-60 we beat the paper at the same recall; with departure delay we tie it on F2; on the day-ahead task we do not beat the paper, whose random split and airport subset are more favourable. Thresholds were set on validation data, so test recall at the "similar recall" operating points lands within about +-3 points of the target and is only approximately matched.

## 10. O6 — Pattern discovery: airport clustering and cancellation association rules (Step 5)
Scripts `mining/05_clustering.py`, `mining/06_association_rules.py`; outputs in `results/clustering/` and `results/association_rules/`. Scope as elsewhere: October excluded, origin airports with known weather.

### 10.1 Airport clustering
**Method.** 113 origin airports with >= 5,000 flights, described by 10 features: log10 flights, % delayed (arrival >= 15 min), mean departure delay, cancellation rate, mean taxi-out, share of flights in moderate/severe weather, weather gap (delayed % in adverse minus clear weather), evening share, late-aircraft share and weather share of delay minutes. Features were standardised and clustered with K-Means (100 restarts). The weather gap is median-imputed for 2 airports with < 150 adverse-weather flights.

**Choice of k.** Silhouette is highest at k = 2 (0.264) and then 0.243 (k = 3), 0.240 (k = 4), 0.176 (k = 5). Taking the best k between 3 and 6 for interpretability gives **k = 3**. Cluster quality is modest: silhouette 0.243 means airports form a continuum with soft boundaries, not sharply separated groups. K-Means agrees with Ward hierarchical clustering (adjusted Rand index 0.82); resampling airports 100 times gives a mean ARI of 0.88 (5th percentile 0.57).

| Cluster | Airports | Mean flights | Delayed % | Dep. delay (min) | Cancel % | Taxi-out (min) | Adverse-weather share % | Weather gap (pp) | Examples |
|---|---|---|---|---|---|---|---|---|---|
| 0 Mid-size, punctual | 35 | 35,846 | 15.9 | 6.8 | 0.94 | 12.7 | 10.3 | 7.6 | PHX, SEA, SLC, FLL, SAN, TPA, PDX, STL |
| 1 Large hubs, congested | 24 | 126,997 | 21.0 | 11.2 | 1.77 | 17.8 | 17.5 | 16.9 | ATL, ORD, DFW, DEN, LAX, SFO, IAH, LAS |
| 2 Smaller, weather-exposed | 54 | 12,220 | 18.1 | 8.7 | 2.33 | 14.2 | 20.2 | 10.7 | BNA, CLE, RDU, MKE, IND, CMH, PIT, PBI |

Reading: the large hubs are the most delayed and the most sensitive to adverse weather (a 16.9 pp jump in the delay rate), with the longest taxi-out times; the mid-size airports are the most reliable and least weather-sensitive; the smaller airports have the highest cancellation rate (2.33%) and the most adverse-weather exposure (20.2% of flights), which is consistent with weather-driven cancellations; the cancellation reasons were not broken down by cluster, so this link is not tested. Files: `cluster_profiles.csv`, `airport_clusters.csv`, `cluster_quality.csv`, `k_selection.csv`, plots `airport_clusters_pca.png`, `airport_clusters_map.png`, `cluster_profile_heatmap.png`.

### 10.2 Cancellation association rules
**Population.** 5,295,135 scheduled flights, 87,076 cancelled (base rate 1.644%, the same 87,076 as Section 3). Each flight is a set of 10 discrete items: month, weekday/weekend, departure period, airline, origin weather class, origin snow, origin high wind, origin freezing (tmax <= 0 C), destination weather class and distance band.

**Part A, exhaustive rules.** All conjunctions of 1–3 items were evaluated exactly (no sampling and no minimum-support approximation) with antecedent >= 2,000 flights and >= 100 cancellations: 6,020 candidate rules. A longer rule is kept only if its confidence is at least 25% higher than every sub-rule it contains. Support, confidence and lift follow the Apriori definitions (lift = P(cancelled | antecedent) / 1.644%).

| Rule (antecedent -> CANCELLED) | Flights | Cancelled | Confidence | Lift |
|---|---|---|---|---|
| airline = MQ & origin snow & destination severe weather | 2,795 | 1,274 | 45.6% | 27.7 |
| origin severe weather & origin high wind | 3,542 | 1,568 | 44.3% | 26.9 |
| origin snow & origin high wind | 3,359 | 1,469 | 43.7% | 26.6 |
| February & airline = MQ & origin severe weather | 2,825 | 1,135 | 40.2% | 24.4 |
| origin severe weather & origin freezing & destination severe weather | 5,475 | 1,848 | 33.8% | 20.5 |
| origin snow & origin severe weather & destination severe weather | 9,833 | 3,232 | 32.9% | 20.0 |
| origin high wind & origin freezing | 5,267 | 1,395 | 26.5% | 16.1 |

The top 20 rules by lift are all weather-driven: severe weather at one or both ends, snow, wind, freezing and February. The regional carrier MQ and (for carrier- and air-system-caused cancellations) EV appear repeatedly: their cancellation rates jump far more than other carriers' when snow or severe weather hits. By reason, rules for weather cancellations reach lifts of 39–43 (about 35–39% of flights covered by the rule are weather-cancelled), and for national-air-system cancellations of 43–54 (EV in snow and severe weather, up to 15% of flights); the highest lifts for airline-caused cancellations are EV with high wind (about 5.5% confidence, lift 12). Files: `top_cancellation_rules.csv`, `rules_by_reason.csv`, `cancellation_rules_all.csv`, `top_cancellation_rules.png`.

**Part B, FP-Growth among the 87,055 cancelled flights** (reason items; security cancellations, 22 flights, left out; min support 8%, lift >= 1.2). The strongest patterns in the cancelled-flight population are: clear weather at both airports and above freezing -> airline-caused cancellation (confidence 55%, lift 1.98), and origin snow -> weather-caused cancellation (confidence 54%, lift 1.28). In other words, cancellations in good weather are mainly carrier decisions, and snow is the strongest single marker of weather cancellations. File: `fpgrowth_rules_among_cancelled.csv`.

**Limitations.** Weather is observed same-day weather at the airport's nearest station, not at the time of the decision. Rules are associations, not causes: MQ and EV rules partly reflect where those carriers operate (northern and eastern airports in winter). Part A only mines conjunctions of up to three items. Part B uses reason labels that DOT assigns by carrier report.
