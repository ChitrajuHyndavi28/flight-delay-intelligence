"""Local dashboard server: serves this folder and a small JSON API backed by the saved T-60 LightGBM model.
   GET  /api/options   valid airlines / routes and form defaults
   POST /api/predict   {airline, origin, dest, month, day_of_week, sched_dep_hour, o_wx, d_wx, prev_status, prev_dep_delay, prev_arr_delay, planned_turn_min, leg_number, holiday}
   GET  /api/sample    a random real Nov-Dec flight (inputs, model probability from its exact features, and what actually happened)
Run:  python dashboard/server.py [port]      (read-only access to the database; never trains)"""
import json, sys, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "mining"))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

bst = lgb.Booster(model_file=str(ROOT / "models" / "lightgbm_t60.txt")); meta = json.load(open(ROOT / "models" / "lightgbm_t60.json"))
FEATS = bst.feature_name(); THR = meta["threshold_sep_max_f1"]
CAT = ["season", "airline", "origin", "dest", "origin_state", "dest_state", "route", "distance_band", "o_wx_class", "d_wx_class", "prev_leg_status"]
print("loading feature table ...", flush=True)
df = pd.read_parquet(ROOT / "mining_cache" / "features_t60.parquet"); df["month"] = df["month"].astype(int)
BASE_RATE = float(df.loc[df.split == "test", "is_delayed"].mean())
last = df.sort_values("date_key").groupby("route").tail(1).set_index("route")
ROUTE = {}
for r, g in df.groupby("route"):
    x = last.loc[r]; ROUTE[r] = dict(origin=g.origin.iat[0], dest=g.dest.iat[0], dist=float(g.distance_miles.median()), mins=float(g.scheduled_time_min.median()), os=g.origin_state.iat[0], ds=g.dest_state.iat[0],
        band=g.distance_band.iat[0], inter=float(g.interstate_flag.iat[0]), hroute=float(x.hist_route_rate), airlines=sorted(g.airline.unique().tolist()))
HAIR = last.groupby("airline").hist_airline_rate.mean().to_dict(); HORI = last.groupby("origin").hist_origin_rate.mean().to_dict()
WXO = [c for c in df.columns if c.startswith("o_") and c not in ("o_wx_class",)]; WXD = [c for c in df.columns if c.startswith("d_") and c not in ("d_wx_class",)]
WX = {}
for side, cols, cls in (("o", WXO, "o_wx_class"), ("d", WXD, "d_wx_class")):
    for (c, m), g in df[df.o_weather_missing == 0].groupby([cls, "month"]): WX[(side, c, m)] = g[cols].median().to_dict()
    for c, g in df[df.o_weather_missing == 0].groupby(cls): WX[(side, c, 0)] = g[cols].median().to_dict()
WOY = df.groupby("month").week_of_year.median().to_dict(); SEASON = df.groupby("month").season.agg(lambda s: s.mode().iat[0]).to_dict()
ORDER = {"first": "no_prev_leg", "no_prev_leg": "no_prev_leg", "not_departed": "not_departed", "airborne": "airborne", "landed": "landed"}
OPTIONS = dict(airlines=sorted(df.airline.unique().tolist()), airports=sorted(df.origin.unique().tolist()), routes={r: dict(origin=v['origin'], dest=v['dest'], airlines=v['airlines']) for r, v in ROUTE.items()}, threshold=THR, base_rate=BASE_RATE)
LABELS = {"prev_leg_dep_delay": "Previous leg left its gate", "prev_leg_arr_delay": "Previous leg arrival delay", "prev_leg_status": "Previous leg status", "planned_turn_min": "Planned turnaround (min)",
          "sched_dep_hour": "Departure hour", "week_of_year": "Time of year", "month": "Month", "o_snow": "Origin snowfall (mm)", "o_prcp": "Origin precipitation (mm)", "d_prcp": "Destination precipitation (mm)", "d_snow": "Destination snowfall (mm)",
          "o_awnd": "Origin wind (m/s)", "d_awnd": "Destination wind (m/s)", "o_wx_class": "Origin weather", "d_wx_class": "Destination weather", "airline": "Airline", "origin": "Origin airport", "dest": "Destination airport", "route": "Route",
          "hist_route_rate": "Route's past delay rate", "hist_airline_rate": "Airline's past delay rate", "hist_origin_rate": "Origin's past delay rate", "leg_number_of_day": "Leg of the day", "day_of_week": "Day of week",
          "sched_arr_hour": "Arrival hour", "prev_leg_sched_arr_minus_t60": "Previous leg's scheduled arrival vs T-60", "distance_miles": "Distance (mi)", "scheduled_time_min": "Scheduled flight time (min)"}
lock = threading.Lock()


def clean(o):
    """JSON for browsers has no NaN/Infinity: turn them into null."""
    if isinstance(o, float): return o if np.isfinite(o) else None
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    return o


def build_row(p):
    route = f"{p['origin']}-{p['dest']}"; ri = ROUTE.get(route)
    if ri is None: raise ValueError(f"No flights were scheduled on {route} in the 2015 data")
    if p["airline"] not in ri["airlines"]: raise ValueError(f"{p['airline']} did not fly {route} in the 2015 data (flown by: {', '.join(ri['airlines'])})")
    month, dow, hr = int(p["month"]), int(p["day_of_week"]), int(p["sched_dep_hour"]); st = ORDER[p["prev_status"]]
    o = WX[("o", p["o_wx"], month)] if ("o", p["o_wx"], month) in WX else WX[("o", p["o_wx"], 0)]; d = WX[("d", p["d_wx"], month)] if ("d", p["d_wx"], month) in WX else WX[("d", p["d_wx"], 0)]
    turn = float(p.get("planned_turn_min") or 45); r = dict(season=SEASON[month], airline=p["airline"], origin=p["origin"], dest=p["dest"], origin_state=ri["os"], dest_state=ri["ds"], route=route, distance_band=ri["band"],
        o_wx_class=p["o_wx"], d_wx_class=p["d_wx"], prev_leg_status=st, month=month, day_of_week=dow, week_of_year=WOY[month], weekend_flag=float(dow >= 6), holiday_flag=float(bool(p.get("holiday"))),
        sched_dep_hour=hr, sched_arr_hour=(hr + int(ri["mins"] // 60)) % 24, distance_miles=ri["dist"], scheduled_time_min=ri["mins"], interstate_flag=ri["inter"],
        prev_leg_arr_delay=np.nan, prev_leg_dep_delay=np.nan, prev_leg_sched_arr_minus_t60=np.nan, planned_turn_min=np.nan, leg_number_of_day=float(p.get("leg_number") or 1),
        hist_route_rate=ri["hroute"], hist_airline_rate=HAIR.get(p["airline"], np.nan), hist_origin_rate=HORI.get(p["origin"], np.nan))
    r.update({k: v for k, v in o.items()}); r.update({k: v for k, v in d.items()})
    if st in ("landed", "airborne"): r["prev_leg_dep_delay"] = float(p.get("prev_dep_delay") or 0)
    if st == "landed": r["prev_leg_arr_delay"] = float(p.get("prev_arr_delay") or 0)
    if st != "no_prev_leg": r["planned_turn_min"] = turn; r["prev_leg_sched_arr_minus_t60"] = 60 - turn
    if st == "no_prev_leg": r["leg_number_of_day"] = 1.0
    return pd.DataFrame([r])


def score(X):
    X = X[FEATS].copy()
    for c in CAT: X[c] = pd.Categorical(X[c])
    for c in FEATS:
        if c not in CAT: X[c] = X[c].astype("float64")
    with lock: prob = float(bst.predict(X)[0]); con = bst.predict(X, pred_contrib=True)[0]
    top = np.argsort(-np.abs(con[:-1]))[:6]; row = X.iloc[0]
    def val(f):
        v = row[f]
        if pd.isna(v): return "not known yet"
        return str(v) if f in CAT else (f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}")
    drivers = [dict(feature=FEATS[j], label=LABELS.get(FEATS[j], FEATS[j]), value=val(FEATS[j]), shap=float(con[j])) for j in top]
    return dict(probability=prob, threshold=THR, predicted_delayed=prob >= THR, base_rate=BASE_RATE, lift=prob / BASE_RATE, drivers=drivers)


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=str(HERE), **k)
    def _json(self, obj, code=200):
        b = json.dumps(clean(obj), allow_nan=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/api/options"): return self._json(OPTIONS)
        if self.path.startswith("/api/sample"): return self._sample()
        return super().do_GET()
    def do_POST(self):
        if not self.path.startswith("/api/predict"): return self._json({"error": "not found"}, 404)
        try:
            p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"); return self._json(score(build_row(p)))
        except Exception as e: return self._json({"error": str(e)}, 400)
    def _sample(self):
        te = df[df.split == "test"]; r = te.iloc[np.random.randint(len(te))]; row = r.to_frame().T
        out = score(row.assign(**{c: row[c] for c in row.columns}))
        from common import connect
        a = connect().execute(f"SELECT arrival_delay_min FROM fact_flight WHERE flight_key = {int(r.flight_key)}").fetchone()[0]
        d = str(int(r.date_key)); st = r.prev_leg_status
        inp = dict(airline=r.airline, origin=r.origin, dest=r.dest, month=int(r.month), day_of_week=int(r.day_of_week), sched_dep_hour=int(r.sched_dep_hour), o_wx=r.o_wx_class, d_wx=r.d_wx_class,
                   prev_status=st, prev_dep_delay=None if pd.isna(r.prev_leg_dep_delay) else float(r.prev_leg_dep_delay), prev_arr_delay=None if pd.isna(r.prev_leg_arr_delay) else float(r.prev_leg_arr_delay),
                   planned_turn_min=None if pd.isna(r.planned_turn_min) else float(r.planned_turn_min), leg_number=int(r.leg_number_of_day), holiday=bool(r.holiday_flag))
        self._json(dict(inputs=inp, date=f"{d[:4]}-{d[4:6]}-{d[6:]}", result=out, actual_arrival_delay_min=float(a), actually_delayed=bool(a >= 15)))
    def log_message(self, *a): pass


if __name__ == "__main__":
    print(f"dashboard on http://localhost:{PORT}/  ({len(ROUTE)} routes, model {bst.num_trees()} trees, threshold {THR:.3f})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
