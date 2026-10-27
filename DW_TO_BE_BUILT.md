# DW to be Built — Flight Delay & Airport Operations Intelligence

Star schema with **FACT_FLIGHT** at the centre and five dimensions: DIM_DATE, DIM_AIRLINE, DIM_AIRPORT (role-played as origin and destination), DIM_ROUTE and DIM_WEATHER. Source columns come from `flights.csv`, `airlines.csv`, `airports.csv`, `2015.csv` (GHCN-Daily) and `ghcnd-stations.txt`.

## 1. Star-schema diagram

```
                         ┌──────────────────┐
                         │    DIM_DATE      │
                         │ PK date_key      │
                         └────────┬─────────┘
                                  │ date_key
┌──────────────────┐     ┌────────┴──────────────────────┐     ┌──────────────────┐
│   DIM_AIRLINE    │     │         FACT_FLIGHT           │     │    DIM_ROUTE     │
│ PK airline_key   ├─────┤ PK flight_key                 ├─────┤ PK route_key     │
└──────────────────┘     │ FK date_key                   │     └──────────────────┘
                         │ FK airline_key                │
┌──────────────────┐     │ FK route_key                  │     ┌──────────────────┐
│   DIM_AIRPORT    │     │ FK origin_airport_key  ───────┼──┐  │   DIM_WEATHER    │
│ PK airport_key   ├──┬──┤ FK destination_airport_key ───┼──┘  │ PK weather_key   │
│ (role-played x2) │  │  │ FK origin_weather_key  ───────┼─────┤ (role-played x2) │
└──────────────────┘  │  │ FK destination_weather_key ───┼─────┤                  │
                      │  │ DD flight_number, tail_number │     └──────────────────┘
                      │  │ measures + indicators         │
                      │  └───────────────────────────────┘
                      └── both airport FKs point to the same DIM_AIRPORT
```

## 2. Table specifications

### FACT_FLIGHT
- **Type:** Transaction fact table.
- **Grain:** one row = one flight operation (one scheduled flight leg) on one date.
- **Primary key:** `flight_key` (surrogate). The natural key is `date_key + airline_key + flight_number + origin_airport_key + scheduled_departure_time`.
- **Foreign keys:**
  - `date_key` → DIM_DATE
  - `airline_key` → DIM_AIRLINE
  - `route_key` → DIM_ROUTE
  - `origin_airport_key` → DIM_AIRPORT
  - `destination_airport_key` → DIM_AIRPORT
  - `origin_weather_key` → DIM_WEATHER
  - `destination_weather_key` → DIM_WEATHER
- **Degenerate dimensions:** `flight_number`, `tail_number`.
- **Time attributes (kept on the fact):** `scheduled_departure_time`, `departure_time`, `wheels_off_time`, `wheels_on_time`, `scheduled_arrival_time`, `arrival_time`.
- **Measures:**

| Measure | Source column | Additivity |
|---|---|---|
| departure_delay_min | DEPARTURE_DELAY | Additive (sum/avg) |
| arrival_delay_min | ARRIVAL_DELAY | Additive |
| air_time_min | AIR_TIME | Additive |
| elapsed_time_min | ELAPSED_TIME | Additive |
| scheduled_time_min | SCHEDULED_TIME | Additive |
| taxi_out_min | TAXI_OUT | Additive |
| taxi_in_min | TAXI_IN | Additive |
| distance_miles | DISTANCE | Additive |
| air_system_delay_min | AIR_SYSTEM_DELAY | Additive |
| security_delay_min | SECURITY_DELAY | Additive |
| airline_delay_min | AIRLINE_DELAY | Additive |
| late_aircraft_delay_min | LATE_AIRCRAFT_DELAY | Additive |
| weather_delay_min | WEATHER_DELAY | Additive |

- **Operational indicators (flags):** `cancelled_flag` (CANCELLED), `diverted_flag` (DIVERTED), `cancellation_reason` (CANCELLATION_REASON: A/B/C/D). Optional derived flags: `departure_delayed_15_flag`, `arrival_delayed_15_flag`.
- **Counting measure:** `flight_count` = 1 per row.

### DIM_DATE
- **Type:** Conformed date dimension (role: flight date).
- **Grain:** one row per calendar day (365 rows for 2015).
- **PK:** `date_key` (integer `YYYYMMDD`). **FKs:** none.
- **Attributes:** `full_date`, `year`, `quarter`, `month_number`, `month_name`, `day_of_month`, `day_of_week_number`, `day_of_week_name`, `week_of_year`, `weekend_flag`, `holiday_flag`, `season`.
- **Source:** `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`.

### DIM_AIRLINE
- **Type:** Descriptive dimension.
- **Grain:** one row per airline (14 rows).
- **PK:** `airline_key` (surrogate). **Natural key:** `airline_iata_code`. **FKs:** none.
- **Attributes:** `airline_iata_code`, `airline_name`.
- **Source:** `airlines.csv` (`IATA_CODE`, `AIRLINE`); joined from `flights.AIRLINE`.

### DIM_AIRPORT
- **Type:** Role-playing dimension (origin and destination).
- **Grain:** one row per airport (322 rows).
- **PK:** `airport_key` (surrogate). **Natural key:** `airport_iata_code`.
- **FKs:** none to other dimensions. Weather association is held in DIM_WEATHER (see section 4), keeping the schema a pure star.
- **Attributes:** `airport_iata_code`, `airport_name`, `city`, `state`, `country`, `latitude`, `longitude`. Optional derived: `nearest_station_id`, `station_distance_km`.
- **Source:** `airports.csv`.

### DIM_ROUTE
- **Type:** Descriptive dimension (origin–destination pair).
- **Grain:** one row per directional origin → destination airport pair.
- **PK:** `route_key` (surrogate). **Natural key:** `origin_iata_code + destination_iata_code`.
- **Attributes:** `origin_iata_code`, `destination_iata_code`, `route_name` (e.g. `LAX-JFK`), `route_distance_miles`, `distance_band` (short/medium/long), `origin_state`, `destination_state`, `interstate_flag`.
- **Optional FKs:** `origin_airport_key` and `destination_airport_key` to DIM_AIRPORT, if the route is not flattened (a snowflake link; the flattened form above is preferred for a strict star).
- **Source:** distinct `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `DISTANCE` from `flights.csv`.

### DIM_WEATHER
- **Type:** Descriptive dimension (daily weather conditions at an airport).
- **Grain:** one row per airport per date, i.e. daily weather from the airport's nearest suitable NOAA station.
- **PK:** `weather_key` (surrogate). **Natural key:** `airport_iata_code + weather_date`.
- **FKs:** none (the fact points to it). The DW can carry `station_id` as an attribute only.
- **Attributes:**
  - `station_id`, `station_name`, `station_latitude`, `station_longitude`, `station_distance_km`
  - `weather_date`
  - `precipitation_tenths_mm` (PRCP), `snowfall_mm` (SNOW), `snow_depth_mm` (SNWD)
  - `tmax_tenths_c` (TMAX), `tmin_tenths_c` (TMIN), `tavg_tenths_c` (TAVG)
  - `avg_wind_speed` (AWND), `peak_gust` (WSF2/WSF5)
  - Derived: `rain_flag`, `snow_flag`, `high_wind_flag`, `weather_severity_category`
- **Source:** `2015.csv` pivoted from one row per element to one column per element, joined to `ghcnd-stations.txt`.
- **Note:** NOAA values come with quality flags. Keep only rows with a blank quality flag; missing values stay NULL.

## 3. DIM_AIRPORT referenced twice (role-playing)

A flight has two airports, so FACT_FLIGHT holds two FKs to the same dimension:
- `origin_airport_key` → DIM_AIRPORT (departure airport)
- `destination_airport_key` → DIM_AIRPORT (arrival airport)

There is one physical table. It is exposed as two views or aliases (`DIM_ORIGIN_AIRPORT`, `DIM_DESTINATION_AIRPORT`) in queries and the BI layer, so a report can filter "departures from ATL" and "arrivals into ORD" at the same time. The same pattern gives `origin_weather_key` and `destination_weather_key` against DIM_WEATHER, which allows departure delays to be explained by origin weather and arrival delays by destination weather.

## 4. Associating NOAA weather stations with airports

`airports.csv` and `ghcnd-stations.txt` share no key, so the link is geographic:

1. **Filter stations:** keep US stations (ID prefix `US`) that have observations in `2015.csv`.
2. **Compute distance:** for each airport, calculate the haversine distance between the airport's `LATITUDE`/`LONGITUDE` and every candidate station's latitude/longitude.
3. **Select the nearest suitable station:** the nearest one that
   - lies within a maximum radius (for example 25–50 km),
   - has data for the needed elements (`PRCP` at minimum, ideally `SNOW`, `TMAX`/`TMIN`, `AWND`),
   - has good daily coverage across 2015.
4. **Fall back by day:** if the nearest station has no value on a given date, use the next-nearest station within the radius and record the `station_id` actually used.
5. **Store the result:** build one DIM_WEATHER row per airport per date with the chosen `station_id` and `station_distance_km`, so the match quality can be audited.
6. **Join in the fact load:** `origin_weather_key` is looked up from (origin airport, flight date), and `destination_weather_key` from (destination airport, flight date).

Airports with no suitable station inside the radius get a default "Unknown weather" member rather than a NULL key.

## 5. DW summary

| Table | Type | Grain | Primary key | Foreign keys | Key attributes / measures |
|---|---|---|---|---|---|
| FACT_FLIGHT | Fact (transaction) | One flight operation on one date | flight_key | date_key, airline_key, route_key, origin_airport_key, destination_airport_key, origin_weather_key, destination_weather_key | Delay, time and distance measures; cancelled/diverted flags; flight_number, tail_number |
| DIM_DATE | Dimension | One calendar day | date_key | – | year, quarter, month, day_of_week, weekend/holiday flags |
| DIM_AIRLINE | Dimension | One airline | airline_key | – | airline_iata_code, airline_name |
| DIM_AIRPORT | Dimension (role-playing) | One airport | airport_key | – | IATA code, name, city, state, country, latitude, longitude |
| DIM_ROUTE | Dimension | One origin → destination pair | route_key | – | origin/destination code, route_name, distance, distance_band |
| DIM_WEATHER | Dimension (role-playing) | One airport per day | weather_key | – | station_id, precipitation, snow, temperature, wind, weather flags |

## 6. Component lists

**Fact table**
- FACT_FLIGHT

**Dimension tables**
- DIM_DATE
- DIM_AIRLINE
- DIM_AIRPORT (origin and destination roles)
- DIM_ROUTE
- DIM_WEATHER (origin and destination roles)

**Measures**
- departure_delay_min, arrival_delay_min
- air_time_min, elapsed_time_min, scheduled_time_min
- taxi_out_min, taxi_in_min
- distance_miles
- air_system_delay_min, security_delay_min, airline_delay_min, late_aircraft_delay_min, weather_delay_min
- flight_count
- Operational indicators: cancelled_flag, diverted_flag (plus cancellation_reason)

**Degenerate dimensions**
- flight_number
- tail_number

## 7. Load notes
- Airport codes in `flights.csv` include 5-digit numeric values that do not exist in `airports.csv`. Map them to an "Unknown airport" member in DIM_AIRPORT, or reject them to an error table.
- Cancelled and diverted flights have NULL time and delay measures. Keep them in the fact so cancellation rates can be computed, and do not treat NULL as 0.
- Convert `hhmm` text times to proper time values; `2400` means midnight.
- `2015.csv` is about 37M rows, so filter to the selected stations and the TMAX, TMIN, TAVG, PRCP, SNOW, SNWD and AWND elements before pivoting.
