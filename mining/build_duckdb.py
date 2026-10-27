"""Build flight_dw.duckdb from the warehouse/*.csv star-schema extracts.

DESTRUCTIVE: this DELETES any existing flight_dw.duckdb and rebuilds the six base tables from CSV. Every derived
table (propagation_features, propagation_t60) is lost, and the file is replaced. It therefore refuses to run if
the database already exists unless --force is given. Derived tables are rebuilt automatically at the end.

    python build_duckdb.py            # first-time build (no database yet)
    python build_duckdb.py --force    # rebuild from scratch (derived tables are dropped, then regenerated)
"""
import argparse, sys
import duckdb
from common import DB_PATH, ROOT

ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true", help="delete and rebuild an existing database")
args = ap.parse_args()
WH = ROOT / "warehouse"

if DB_PATH.exists() and not args.force:
    sys.exit(f"REFUSING TO RUN: {DB_PATH} already exists.\n"
             f"Rebuilding deletes the file and every derived table (propagation_features, propagation_t60); other scripts that are "
             f"using the database would also break.\nRe-run with --force if you really want to rebuild.")
if DB_PATH.exists():
    print(f"WARNING: --force given. Deleting {DB_PATH}; derived tables (propagation_features, propagation_t60) will be LOST "
          f"and then regenerated. Backups, if any, are in backups/.", flush=True)
    DB_PATH.unlink()
    wal = DB_PATH.with_suffix(".duckdb.wal")
    if wal.exists(): wal.unlink()

con = duckdb.connect(str(DB_PATH))
for t in ["dim_date", "dim_airline", "dim_airport", "dim_route", "dim_weather", "fact_flight"]:
    con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{WH / t}.csv', header=true, sample_size=-1)")
    print(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
con.execute("CHECKPOINT"); con.close()

from common import ensure_propagation_table, ensure_propagation_t60      # regenerate derived tables so the DB is complete
ensure_propagation_table(force=True); ensure_propagation_t60(force=True)
print("derived tables rebuilt: propagation_features, propagation_t60")
