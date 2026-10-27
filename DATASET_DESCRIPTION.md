# Dataset Description — Flight Delay & Airport Operations Intelligence

All data lives in `datasets/` and covers **calendar year 2015**.

| File | Role | Size / Rows |
|---|---|---|
| `flights.csv` | Flight operations (fact table) | ~592 MB, 5,819,079 flights |
| `airlines.csv` | Airline lookup | 14 airlines |
| `airports.csv` | Airport lookup with coordinates | 322 airports |
| `2015.csv` | NOAA GHCN-Daily weather observations | ~1.3 GB, 36,914,550 rows |
| `ghcnd-stations.txt` | NOAA weather station metadata | 133,039 stations (fixed-width) |

## 1. Flight data — `flights.csv`
One row per scheduled US domestic flight in 2015 (all rows have `YEAR = 2015`). It is the core dataset for delay and operations analysis.

**Columns (31):**
- **Date:** `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`
- **Flight identity:** `AIRLINE` (IATA code), `FLIGHT_NUMBER`, `TAIL_NUMBER`
- **Route:** `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `DISTANCE`
- **Departure:** `SCHEDULED_DEPARTURE`, `DEPARTURE_TIME`, `DEPARTURE_DELAY`, `TAXI_OUT`, `WHEELS_OFF`
- **In flight / arrival:** `SCHEDULED_TIME`, `ELAPSED_TIME`, `AIR_TIME`, `WHEELS_ON`, `TAXI_IN`, `SCHEDULED_ARRIVAL`, `ARRIVAL_TIME`, `ARRIVAL_DELAY`
- **Status:** `DIVERTED`, `CANCELLED`, `CANCELLATION_REASON`
- **Delay causes (minutes):** `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`

**Purpose:** measure on-time performance, delay and cancellation rates, taxi times, and the share of delay attributed to each cause, by airline, route, airport, and time of day/week/month.

**Data quality notes:** times are stored as `hhmm` strings; delay-cause columns are only populated for delayed flights; cancelled/diverted flights have blank time fields; `ORIGIN_AIRPORT` / `DESTINATION_AIRPORT` contain 3-letter IATA codes **and** some 5-digit numeric codes (e.g. `10135`, `10140`) that do not match `airports.csv` and must be handled or filtered.

## 2. Airline data — `airlines.csv`
Lookup of 14 carriers. **Columns:** `IATA_CODE`, `AIRLINE` (full name), e.g. `UA` → United Air Lines Inc., `WN` → Southwest Airlines Co.

**Purpose:** converts `flights.AIRLINE` codes into readable names for airline-level comparisons and dashboards.

## 3. Airport data — `airports.csv`
Lookup of 322 airports. **Columns:** `IATA_CODE`, `AIRPORT`, `CITY`, `STATE`, `COUNTRY`, `LATITUDE`, `LONGITUDE`.

**Purpose:** gives airport names and locations for geographic analysis and maps, and supplies the coordinates needed to link each airport to nearby weather stations.

## 4. NOAA weather data — `2015.csv` and `ghcnd-stations.txt`
Both are from NOAA's Global Historical Climatology Network–Daily (GHCN-Daily).

**`2015.csv`** — daily observations, one row per station / date / element. The file has **no header row**; the standard GHCN-Daily layout is:
`station ID, date (YYYYMMDD), element, value, measurement flag, quality flag, source flag, observation time`.
Elements seen in the file include `PRCP` (precipitation), `TMAX`, `TMIN`, `TAVG` (temperature), `SNOW`, `SNWD` (snowfall / snow depth), `WESD`, `AWND` (average wind), `WSF2`, `WSF5` (peak gusts), and `TOBS`. The data spans 1 Jan – 31 Dec 2015 and is global (not only US stations).

**`ghcnd-stations.txt`** — fixed-width station catalogue (no header): station ID, latitude, longitude, elevation, state, station name, plus GSN/HCN-CRN and WMO flags. The ID's first two characters are the country code, so US stations can be filtered (`US…`).

**Purpose:** explain weather-related delays by adding precipitation, snow, temperature, and wind conditions at each airport on each day.

## How the datasets are integrated

```
airlines.csv ──(IATA_CODE = AIRLINE)──┐
                                      ├── flights.csv ◄── (airport, date) ◄── weather
airports.csv ──(IATA_CODE = ORIGIN_/DESTINATION_AIRPORT)─┘
```

1. **Flights ↔ Airlines:** join `flights.AIRLINE` to `airlines.IATA_CODE`.
2. **Flights ↔ Airports:** join `ORIGIN_AIRPORT` and `DESTINATION_AIRPORT` to `airports.IATA_CODE` (twice, once per role) to get names, city, state, and coordinates. Numeric airport codes won't match and should be excluded or mapped.
3. **Airports ↔ Weather stations:** `ghcnd-stations.txt` and `airports.csv` share no key, so link them **spatially** — for each airport, pick the nearest station(s) by `LATITUDE`/`LONGITUDE` (haversine distance, with a distance cutoff), preferably stations reporting the needed elements.
4. **Weather observations ↔ Stations:** join `2015.csv` station ID to `ghcnd-stations.txt` ID, then pivot `ELEMENT`/`VALUE` into one column per variable per station-day.
5. **Weather ↔ Flights:** build a date from `YEAR`, `MONTH`, `DAY` in `flights.csv` and join with the weather station-day on (nearest station for the origin airport, date). Origin weather explains departure delays; destination weather can be joined the same way for arrival delays.

The result is a flight-level table enriched with airline name, airport details, and daily weather at the origin (and optionally destination), used for delay analysis, airport operations metrics, and weather-impact modelling. `WEATHER_DELAY` in `flights.csv` can serve as a sanity check on the weather link.

**Caveats:** weather is daily, not hourly, so it cannot be matched to the specific departure time; station coverage and element availability vary by location; both large files (`flights.csv`, `2015.csv`) should be processed in chunks or with a tool like DuckDB/Spark.
