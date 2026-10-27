# Sample OLAP Operations — Flight Delay & Airport Operations Intelligence

All queries run on the star schema in `DW_TO_BE_BUILT.md`: `FACT_FLIGHT` joined to `DIM_DATE`, `DIM_AIRLINE` and `DIM_AIRPORT` (as origin). Every value in the result tables is a **placeholder** (`‹…›`); no real results are shown.

## Metric definitions

| Metric | Expression |
|---|---|
| Flight count | `COUNT(*)` (or `SUM(flight_count)`) |
| Avg departure delay | `AVG(departure_delay_min)` |
| Avg arrival delay | `AVG(arrival_delay_min)` |
| Cancellation count | `SUM(cancelled_flag)` |
| Cancellation rate | `SUM(cancelled_flag) * 1.0 / COUNT(*)` |
| On-time % | flights with `arrival_delay_min <= 15`, divided by flights that were neither cancelled nor diverted |

`AVG` ignores NULLs, so cancelled and diverted flights (which have no delay values) do not distort the delay averages.

---

## 1. Roll-Up: Day → Month → Quarter → Year

**Why it is a Roll-Up:** the data moves *up* the DIM_DATE hierarchy from fine to coarse. Each step drops a lower level from the `GROUP BY` and the measures are re-aggregated at the higher level.

**Level 1: Day (lowest level)**
```sql
SELECT d.full_date,
       COUNT(*)                     AS flight_count,
       AVG(f.departure_delay_min)   AS avg_departure_delay,
       AVG(f.arrival_delay_min)     AS avg_arrival_delay,
       SUM(f.cancelled_flag)        AS cancellation_count
FROM fact_flight f JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.full_date;
```

**Level 2: Month** (remove `day`)
```sql
SELECT d.year, d.month_number, d.month_name,
       COUNT(*), AVG(f.departure_delay_min), AVG(f.arrival_delay_min), SUM(f.cancelled_flag)
FROM fact_flight f JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.month_number, d.month_name;
```

**Level 3: Quarter** (remove `month`)
```sql
SELECT d.year, d.quarter, COUNT(*), AVG(f.departure_delay_min),
       AVG(f.arrival_delay_min), SUM(f.cancelled_flag)
FROM fact_flight f JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.quarter;
```

**Level 4: Year** (remove `quarter`)
```sql
SELECT d.year, COUNT(*), AVG(f.departure_delay_min),
       AVG(f.arrival_delay_min), SUM(f.cancelled_flag)
FROM fact_flight f JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year;
```

**Result structure**

| Level | Period | Flight count | Avg departure delay (min) | Avg arrival delay (min) | Cancellation count |
|---|---|---|---|---|---|
| Day | ‹YYYY-MM-DD› | ‹value› | ‹value› | ‹value› | ‹value› |
| Month | ‹Month YYYY› | ‹value› | ‹value› | ‹value› | ‹value› |
| Quarter | ‹Qn YYYY› | ‹value› | ‹value› | ‹value› | ‹value› |
| Year | ‹YYYY› | ‹value› | ‹value› | ‹value› | ‹value› |

**Insight:** the detailed view shows daily volatility, such as storm days. The higher levels smooth it into seasonal and annual trends. Flight count and cancellation count are additive, so they roll up by summing. The averages must be recomputed from the base rows (or from sums and counts); averaging the daily averages would be wrong because days have different flight counts.

---

## 2. Drill-Down: Year → Quarter → Month → Day (2015)

**Why it is a Drill-Down:** the analysis starts at the highest level (Year = 2015) and moves *down* the same hierarchy into more detail, adding one lower level at a time and fixing the parent value. It is the reverse of Roll-Up.

**Step 1: Year**
```sql
SELECT d.year, COUNT(*) AS flight_count,
       (AVG(f.departure_delay_min) + AVG(f.arrival_delay_min)) / 2 AS avg_delay,  -- or report each separately
       SUM(f.cancelled_flag) AS cancellation_count
FROM fact_flight f JOIN dim_date d ON f.date_key = d.date_key
WHERE d.year = 2015 GROUP BY d.year;
```

**Step 2: Quarter** (add `quarter`)
```sql
... WHERE d.year = 2015 GROUP BY d.year, d.quarter;
```

**Step 3: Month** (add `month`, e.g. after choosing Q1)
```sql
... WHERE d.year = 2015 AND d.quarter = ‹selected quarter› GROUP BY d.year, d.quarter, d.month_number, d.month_name;
```

**Step 4: Day** (add `day`, e.g. after choosing a month)
```sql
... WHERE d.year = 2015 AND d.month_number = ‹selected month› GROUP BY d.full_date;
```

"Average delay" can be reported as average arrival delay (`AVG(arrival_delay_min)`), or as departure and arrival side by side. Pick one and keep it the same at every step.

**Result structure**

| Drill step | Period | Flight count | Avg delay (min) | Cancellation count |
|---|---|---|---|---|
| 1 | 2015 | ‹value› | ‹value› | ‹value› |
| 2 | ‹Q1 / Q2 / Q3 / Q4› | ‹value› | ‹value› | ‹value› |
| 3 | ‹Month in selected quarter› | ‹value› | ‹value› | ‹value› |
| 4 | ‹Date in selected month› | ‹value› | ‹value› | ‹value› |

**Insight:** a high-level result (for example, one quarter with a higher delay) is traced to the month and then the specific days that cause it, which helps locate root causes such as weather events.

---

## 3. Slice: Airline = Delta Air Lines

**Why it is a Slice:** a single condition on **one dimension** (DIM_AIRLINE) fixes that dimension to one value. The result is a "sub-cube" for Delta, and all other dimensions remain free.

```sql
SELECT COUNT(*)                                                             AS total_flights,
       AVG(f.departure_delay_min)                                           AS avg_departure_delay,
       AVG(f.arrival_delay_min)                                             AS avg_arrival_delay,
       SUM(f.cancelled_flag) * 100.0 / COUNT(*)                             AS cancellation_rate_pct,
       SUM(CASE WHEN f.arrival_delay_min <= 15 THEN 1 ELSE 0 END) * 100.0
         / NULLIF(SUM(CASE WHEN f.cancelled_flag = 0 AND f.diverted_flag = 0 THEN 1 ELSE 0 END), 0)
                                                                            AS on_time_pct
FROM fact_flight f JOIN dim_airline a ON f.airline_key = a.airline_key
WHERE a.airline_name = 'Delta Air Lines Inc.';   -- IATA code DL
```

| Airline | Total flights | Avg departure delay (min) | Avg arrival delay (min) | Cancellation rate | On-time % |
|---|---|---|---|---|---|
| Delta Air Lines Inc. | ‹value› | ‹value› | ‹value› | ‹value›% | ‹value›% |

**Insight:** shows Delta's overall reliability, which can then be compared with the other airlines' slices.

---

## 4. Dice: Year = 2015, Quarter = Q1, Airline = Delta, Origin State = New York

**Why it is a Dice:** conditions are applied on **several dimensions at once** (DIM_DATE: year and quarter; DIM_AIRLINE: airline; DIM_AIRPORT as origin: state). The result is a smaller sub-cube defined by value ranges on multiple dimensions, unlike a slice, which fixes only one.

```sql
SELECT COUNT(*)                                  AS flight_count,
       AVG(f.departure_delay_min)                AS avg_departure_delay,
       AVG(f.arrival_delay_min)                  AS avg_arrival_delay,
       SUM(f.cancelled_flag) * 100.0 / COUNT(*)  AS cancellation_rate_pct
FROM fact_flight f
JOIN dim_date    d  ON f.date_key = d.date_key
JOIN dim_airline a  ON f.airline_key = a.airline_key
JOIN dim_airport o  ON f.origin_airport_key = o.airport_key      -- origin role
WHERE d.year = 2015
  AND d.quarter = 1                                               -- Q1
  AND a.airline_name = 'Delta Air Lines Inc.'
  AND o.state = 'NY';                                             -- New York
```

| Year | Quarter | Airline | Origin state | Flight count | Avg departure delay (min) | Avg arrival delay (min) | Cancellation rate |
|---|---|---|---|---|---|---|---|
| 2015 | Q1 | Delta Air Lines Inc. | NY | ‹value› | ‹value› | ‹value› | ‹value›% |

**Insight:** isolates Delta's winter-quarter performance out of New York airports, a segment where weather is expected to matter.

---

## 5. Pivot: Airline (rows) × Month (columns) → Average Arrival Delay

**Why it is a Pivot:** it does not filter or change the level of detail. It **rotates the axes** so that the airline dimension runs down the rows and the month dimension across the columns, with one measure in the cells.

```sql
SELECT a.airline_name,
       AVG(CASE WHEN d.month_number = 1  THEN f.arrival_delay_min END) AS jan,
       AVG(CASE WHEN d.month_number = 2  THEN f.arrival_delay_min END) AS feb,
       -- ... repeat for months 3 to 11 ...
       AVG(CASE WHEN d.month_number = 12 THEN f.arrival_delay_min END) AS dec
FROM fact_flight f
JOIN dim_airline a ON f.airline_key = a.airline_key
JOIN dim_date    d ON f.date_key = d.date_key
WHERE d.year = 2015
GROUP BY a.airline_name;
```

| Airline | Jan | Feb | Mar | … | Nov | Dec |
|---|---|---|---|---|---|---|
| ‹Airline 1› | ‹value› | ‹value› | ‹value› | … | ‹value› | ‹value› |
| ‹Airline 2› | ‹value› | ‹value› | ‹value› | … | ‹value› | ‹value› |
| ‹Airline n› | ‹value› | ‹value› | ‹value› | … | ‹value› | ‹value› |

**Insight to draw once the cells are filled:**
- Reading across a row shows each airline's seasonality, such as which months are its worst.
- Reading down a column shows which airlines perform best or worst in a given month.
- Cells that stand out from both their row and column point to airline-specific problems, as opposed to industry-wide seasonal effects.

---

## Summary of operations

| Operation | What changes | Dimension(s) involved | Distinguishing feature |
|---|---|---|---|
| Roll-Up | Aggregates upward in the hierarchy | DIM_DATE | Fine → coarse (Day → Year) |
| Drill-Down | Adds detail downward in the hierarchy | DIM_DATE | Coarse → fine (Year → Day) |
| Slice | Fixes one dimension to one value | DIM_AIRLINE | One filter, one dimension |
| Dice | Restricts several dimensions | DIM_DATE, DIM_AIRLINE, DIM_AIRPORT (origin) | Multiple filters, multiple dimensions |
| Pivot | Rotates the view | DIM_AIRLINE × DIM_DATE | Same data, new orientation |
