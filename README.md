# Flight Delay & Airport Operations Intelligence

Data warehouse (DuckDB star schema, 5,819,079 flights from the 2015 US DOT on-time data + NOAA GHCN-Daily weather) with an analysis and mining layer: OLAP export, root-cause and weather impact, delay propagation, and predictive mining (LightGBM vs baselines). The full write-up is `REPORT.md`.

## Status
| Step | Script | Status |
|---|---|---|
| Build DB | `mining/build_duckdb.py` | done |
| 0 Audit | `mining/00_audit.py` | done |
| 1 OLAP export (O2) | `mining/01_olap_export.py` | done |
| 2 Root cause and weather (O3) | `mining/02_root_cause_weather.py` | done |
| 3 Delay propagation (O4) | `mining/03_delay_propagation.py` | done |
| 4 Predictive mining (O5) | `mining/04_features.sql`, `mining/04_train_compare.py`, `mining/04_explain.py` | done |
| 4c Neural network + ensemble | `mining/07_neural_net.py` | done |
| 4d Departure-delay scenario (base-paper comparison) | `mining/08_post_departure.py` | done |
| 4e Particle-swarm tuning + weather ON/OFF | `mining/09_pso_tuning_final.py`, `mining/pso.py` | done |
| 5 Clustering and association rules (O6) | `mining/05_clustering.py`, `mining/06_association_rules.py` | done |
| Key Insights list in REPORT.md | | **not yet written** |
| Multi-page portal + live predictor (local web page) | `bash run_dashboard.sh` -> http://localhost:8765. Pages: Overview, T-60, Day-ahead, Post-departure, Metric evaluation, Explore data, About. Files in `dashboard/` (`server.py`, `index.html`, `style.css`, `lib.js`, `pages.js`, `main.js`); data built by `mining/11_dashboard_data.py` | done |
| Project report in the university template (Word + PDF) | `mining/14_build_project_report.py` -> `PROJECT_REPORT.docx` / `PROJECT_REPORT.pdf` (diagrams: `mining/13_report_diagrams.py`; page numbers: `mining/15_report_page_map.py`) | done |
| Final report (Word) | `mining/10_build_final_report.py` -> `FINAL_REPORT.docx` | done |
| Live demo | `mining/demo_live.py` | done, tested twice |

## Setup
Python 3.9 virtual environment in `.venv` with the packages pinned in `requirements.txt` (plus PyTorch for `07_neural_net.py`: `.venv/bin/pip install torch`). LightGBM and XGBoost need the OpenMP runtime on macOS.

```
brew install libomp
.venv/bin/pip install -r requirements.txt
```

## Rebuild the data (files not stored in this repository)
The raw data and the databases are too large for GitHub (files over 100 MB) and are listed in `.gitignore`: `datasets/`, `warehouse/fact_flight.csv`, `warehouse/*.db`, `flight_dw.duckdb`, `backups/` and `mining_cache/`. Everything else (code, results, models, the portal and the reports) is in the repository. To rebuild the data:

1. **Download the raw files** into `datasets/`:
   - `flights.csv`, `airlines.csv`, `airports.csv`: the 2015 US DOT "Flight Delays and Cancellations" dataset (Kaggle: `usdot/flight-delays`, originally from the Bureau of Transportation Statistics).
   - `2015.csv`: NOAA GHCN-Daily for 2015, `https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2015.csv.gz` (unzip it).
   - `ghcnd-stations.txt`: `https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt`.
2. **Set up the environment** (see Setup above).
3. **Build the warehouse CSVs** (about 5.8 million fact rows): `.venv/bin/python etl_pipeline.py`
4. **Load DuckDB**: `.venv/bin/python mining/build_duckdb.py` (first build; it refuses to overwrite an existing database unless `--force` is given). The derived tables are rebuilt automatically.
5. **Run the analysis and models** with the commands in "Reproduce the analysis". Model predictions and the feature table are cached in `mining_cache/` and regenerate automatically.
6. **Start the portal**: `bash run_dashboard.sh`, then open http://localhost:8765. The saved T-60 model (`models/lightgbm_t60.txt`) is in the repository; the live predictor also needs the feature table in `mining_cache/`, which step 5 builds.

## Reproduce the analysis
Run from the project root. Every script reads `flight_dw.duckdb` and writes to `results/`.

```
.venv/bin/python mining/00_audit.py
.venv/bin/python mining/01_olap_export.py
.venv/bin/python mining/02_root_cause_weather.py
.venv/bin/python mining/03_delay_propagation.py
.venv/bin/python mining/04a_t60_spot_check.py
bash mining/run_step4.sh
bash mining/run_step4b.sh
.venv/bin/python mining/04_explain.py
.venv/bin/python mining/07_neural_net.py --scenarios t60 dayahead [--tune]
.venv/bin/python mining/08_post_departure.py
.venv/bin/python mining/09_pso_tuning_final.py
.venv/bin/python mining/05_clustering.py
.venv/bin/python mining/06_association_rules.py
.venv/bin/python mining/make_step4_tables.py
```

`run_step4.sh` and `run_step4b.sh` train the models (roughly one to two hours in total on a 10-core machine; every model's predictions are cached in `mining_cache/preds/`, so an interrupted run resumes where it stopped). `run_step4b.sh` waits for `run_step4.sh` to finish.

## Database safety (read this before touching `flight_dw.duckdb`)
- `mining/build_duckdb.py` **deletes the database file** and rebuilds the six base tables from `warehouse/*.csv`. It now refuses to run if `flight_dw.duckdb` exists unless `--force` is given, prints a warning that the derived tables are lost, and regenerates them at the end.
- The two derived tables, `propagation_features` (Step 3) and `propagation_t60` (Step 4), are rebuilt automatically by any script that needs them (`common.ensure_propagation_table`, `common.ensure_propagation_t60`). They take a few seconds. Existence is checked on a read-only connection; a write connection is opened only if a table is missing.
- Audit, OLAP, analysis, training, explainability and demo scripts open DuckDB with `read_only=True`. Only `build_duckdb.py` and the automatic table rebuild write to the file. A writer cannot open the file while another process holds it, and the rebuild then stops with a clear error instead of corrupting anything.
- Set the environment variable `FLIGHT_DB` to point every script at a different database file (used to test the rebuild paths on a scratch copy).

### Backup
`backups/flight_dw_final.duckdb` is a verified copy of the final database (base tables and both derived tables). Its SHA-256 is in `backups/flight_dw_final.duckdb.sha256`:

```
53bdbd45724844e057b01432e657f25e4283d798cbc87c1f12676806eefbbc87
```

Check the checksum and restore if the live file is damaged or missing tables:

```
shasum -a 256 backups/flight_dw_final.duckdb
cp backups/flight_dw_final.duckdb flight_dw.duckdb
```

### Incident record: derived tables missing (2026-10-13)
- **What happened.** While resuming the second training wave, the first stage failed with `Table with name propagation_t60 does not exist`. Both derived tables were gone from `flight_dw.duckdb`; the six base tables were intact (5,819,079 fact rows).
- **Likely cause.** The old `mining/build_duckdb.py` called `DB_PATH.unlink()` unconditionally and then recreated only the six base tables. That leaves exactly this state (base tables present, derived tables missing). The project owner recalled possibly running scripts, including `build_duckdb.py`, from the VS Code terminal before the tables disappeared. This was **not confirmed** (no shell history or logs were inspected). Stopping the background jobs shortly before is an unlikely cause, because DuckDB checkpoints tables to disk, but it cannot be fully excluded.
- **Impact.** Derived tables lost. No result, cached prediction or model was affected.
- **Recovery and verification.** Both tables were regenerated from SQL. The rebuilt tables matched the earlier ones exactly (status counts 2,395,651 airborne / 1,846,809 no predecessor / 1,458,836 landed / 103,062 not departed; 3,917,656 `prev_ok`; 3,693,636 `feat_ok`), and the regenerated feature matrix equalled the cached parquet cell for cell.
- **Prevention.** `build_duckdb.py` now needs `--force`; derived tables rebuild themselves; read-only connections are the default; a verified backup exists.

## Live demo
Prepared for Review 2: justify the algorithm and show the code running. About 40 seconds end to end on this machine, no internet needed, read-only database access, never trains the full model.

```
cd /Users/hyndavi/Desktop/flight
source .venv/bin/activate
python mining/demo_live.py --stage 1
python mining/demo_live.py --stage 2
python mining/demo_live.py --stage 3
python mining/demo_live.py --stage all
```

- Stage 1 shows the warehouse (row counts, a quarterly roll-up and an airline x month pivot, each with its query time).
- Stage 2 trains nine models live on 200,000 training flights and scores 100,000 test flights, with the hyperparameters chosen in Step 4 and no tuning. It prints a ranked table and saves it to `results/demo/live_comparison.csv`. The full-data table is `results/model_comparison/comparison_table.csv`.
- Stage 3 scores five real November–December flights with the saved full-data model `models/lightgbm_t60.txt`, with the top SHAP drivers per flight.
- Every run writes `results/demo/demo_run_<timestamp>.txt` (a backup of exactly what was shown).
- If a stage fails it prints a short message instead of a traceback and writes the details to `results/demo/demo_error_*.txt`.

What to expect and say: in the full-data study LightGBM and tuned XGBoost are statistically tied (PR-AUC about 0.639 each), and the live sample may show either one first; the demo prints a note when the top two are closer than 0.01. Stage 3 uses a fixed random seed, so the same five flights appear every time, including any it gets wrong.

Before presenting, close other terminals or notebooks that have `flight_dw.duckdb` open for writing.

## Layout
- `warehouse/` star-schema CSV extracts (only the small dimension files are in the repository); `flight_dw.duckdb` the database; `backups/` the verified copy.
- `mining/` all scripts (`common.py` and `ml_common.py` hold shared helpers: connection, plotting, weather classes, leakage assertion, preprocessing and model definitions).
- `results/` per-step tables and charts; `results/model_comparison/` and `results/explain/` hold Step 4; `results/demo/` the demo outputs.
- `models/` the saved full-data T-60 LightGBM model and its metadata; `mining_cache/` feature matrix and cached predictions (regenerated by the scripts).
