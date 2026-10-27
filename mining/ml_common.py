"""Shared pieces for Step 4: feature groups, leakage assertion, data loading, metrics, threshold tuning, bootstrap."""
import json, re, time
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from common import ROOT, SEED, connect, ensure_propagation_t60

CACHE = ROOT / "mining_cache"; CACHE.mkdir(exist_ok=True)
FEATURE_FILE = CACHE / "features_t60.parquet"
TARGET = "is_delayed"

CAT_COLS = ["season", "airline", "origin", "dest", "origin_state", "dest_state", "route", "distance_band",
            "o_wx_class", "d_wx_class", "prev_leg_status"]
WX_NUM = ["o_prcp", "o_snow", "o_snwd", "o_tmax", "o_tmin", "o_awnd", "o_weather_missing",
          "d_prcp", "d_snow", "d_snwd", "d_tmax", "d_tmin", "d_awnd", "d_weather_missing"]
WX_CAT = ["o_wx_class", "d_wx_class"]
PROP_LIVE = ["prev_leg_status", "prev_leg_arr_delay", "prev_leg_dep_delay"]                     # status of predecessor at T-60
PROP_SCHED = ["prev_leg_sched_arr_minus_t60", "planned_turn_min", "leg_number_of_day"]            # schedule-known structure
PROP_ALL = PROP_LIVE + PROP_SCHED
HIST = ["hist_route_rate", "hist_airline_rate", "hist_origin_rate"]
NUM_COLS = (["month", "day_of_week", "week_of_year", "weekend_flag", "holiday_flag", "sched_dep_hour", "sched_arr_hour",
             "distance_miles", "scheduled_time_min", "interstate_flag"] + WX_NUM
            + ["prev_leg_arr_delay", "prev_leg_dep_delay", "prev_leg_sched_arr_minus_t60", "planned_turn_min", "leg_number_of_day"] + HIST)
ALL_FEATURES = CAT_COLS + NUM_COLS
NON_FEATURE_COLS = ["flight_key", "date_key", "split", TARGET]

# ---- scenarios (feature sets) -------------------------------------------------------------------------------
SCENARIOS = {
    "t60": ALL_FEATURES,                                                                         # main: operational model
    "dayahead": [c for c in ALL_FEATURES if c not in PROP_ALL],                                  # no propagation features at all
    "dayahead_plus_sched": [c for c in ALL_FEATURES if c not in PROP_LIVE],                      # + schedule-known turn structure
    "t60_no_weather": [c for c in ALL_FEATURES if c not in WX_NUM + WX_CAT],
    "t60_no_month": [c for c in ALL_FEATURES if c not in ["month", "week_of_year"]],
    "t60_no_hist": [c for c in ALL_FEATURES if c not in HIST],
    "dayahead_no_weather": [c for c in ALL_FEATURES if c not in PROP_ALL + WX_NUM + WX_CAT],
}

# ---- leakage assertion ---------------------------------------------------------------------------------------
FORBIDDEN_EXACT = {"departure_delay_min", "arrival_delay_min", "departure_min", "arrival_min", "wheels_off_min", "wheels_on_min",
                   "taxi_out_min", "taxi_in_min", "air_time_min", "elapsed_time_min", "air_system_delay_min", "security_delay_min",
                   "airline_delay_min", "late_aircraft_delay_min", "weather_delay_min", "cancellation_reason", "cancelled_flag",
                   "diverted_flag", "departure_delayed_15_flag", "arrival_delayed_15_flag", "is_delayed", "t60_abs", "prev_arr_abs",
                   "prev_out_abs", "sdep_abs", "legacy_prev_leg_arr_delay", "legacy_turnaround_min", "flight_key", "date_key", "tail_number"}
FORBIDDEN_PATTERN = re.compile(r"(taxi|air_time|elapsed|wheels|cancel|divert|^departure_|^arrival_|_delay_min$|^actual|legacy)")
ALLOWED_PREV = {"prev_leg_arr_delay", "prev_leg_dep_delay", "prev_leg_sched_arr_minus_t60"}   # predecessor's T-60 state (explicitly allowed)


def assert_no_leakage(features, verbose=True, allow_rebuild=True):
    """Fails if any forbidden/leaky column is in the feature list, if a column is outside the approved whitelist, and
    (against the DB) if any predecessor-derived value uses an actual time later than T-60."""
    feats = list(features)
    bad = [c for c in feats if c in FORBIDDEN_EXACT or (FORBIDDEN_PATTERN.search(c) and c not in ALLOWED_PREV)]
    assert not bad, f"LEAKAGE: forbidden columns in feature matrix: {bad}"
    unknown = [c for c in feats if c not in ALL_FEATURES]
    assert not unknown, f"Features outside the approved whitelist: {unknown}"
    assert len(set(feats)) == len(feats), "duplicate features"
    con = connect()                               # read-only
    has_table = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='propagation_t60'").fetchone()[0] > 0
    if not has_table:
        if not allow_rebuild:                     # e.g. the live demo: must never write to the database
            if verbose: print(f"[leakage assertion] OK (static checks): {len(feats)} features, none forbidden; DB-side T-60 check skipped (propagation_t60 missing, rebuild not allowed here)")
            return
        con.close(); ensure_propagation_t60(); con = connect()      # rebuild automatically if the derived table is missing
    r = con.execute("""SELECT sum((prev_leg_arr_delay IS NOT NULL AND (prev_arr_abs > t60_abs OR prev_leg_status <> 'landed'))::int),
        sum((prev_leg_dep_delay IS NOT NULL AND (prev_out_abs > t60_abs OR prev_leg_status NOT IN ('landed','airborne')))::int),
        sum((prev_leg_status='landed' AND prev_arr_abs > t60_abs)::int), sum((prev_leg_status='airborne' AND (prev_out_abs > t60_abs OR prev_arr_abs <= t60_abs))::int),
        sum((prev_leg_status='not_departed' AND prev_out_abs <= t60_abs)::int) FROM propagation_t60""").fetchone()
    assert all(v == 0 for v in r), f"LEAKAGE: propagation feature uses an actual time later than T-60: {r}"
    if verbose: print(f"[leakage assertion] OK: {len(feats)} features, none forbidden; 0 predecessor values use an event after T-60")


# ---- data ------------------------------------------------------------------------------------------------------
def build_features(force=False):
    """Run 04_features.sql (single query over the star schema) -> parquet cache."""
    if FEATURE_FILE.exists() and not force: return
    ensure_propagation_t60()
    con = connect(); sql = (ROOT / "mining" / "04_features.sql").read_text()
    con.execute(f"COPY ({sql}) TO '{FEATURE_FILE}' (FORMAT PARQUET)")


def load_features(columns=None):
    build_features()
    df = pd.read_parquet(FEATURE_FILE)
    for c in CAT_COLS: df[c] = df[c].astype("category")
    for c in NUM_COLS: df[c] = df[c].astype("float32")
    return df


def split_masks(df):
    return df.split.eq("train").to_numpy(), df.split.eq("val").to_numpy(), df.split.eq("test").to_numpy()


def class_weights(y):
    """Balanced weights, identical for every model: w_c = n / (2 * n_c) computed on the TRAINING set."""
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    return {0: n / (2 * n0), 1: n / (2 * n1)}


# ---- metrics -----------------------------------------------------------------------------------------------------
def best_f1_threshold(y_val, s_val):
    p, r, t = precision_recall_curve(y_val, s_val)
    f = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.argmax(f)); return float(t[i]), float(f[i])


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    return dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s), f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                accuracy=accuracy_score(y, pred), predicted_positive_rate=pred.mean())


class PoissonBootstrapAP:
    """Fast paired bootstrap of PR-AUC ('ap', sklearn step definition incl. ties) and ROC-AUC ('auc') using Poisson(1)
    resampling weights. The SAME weights are reused for every model, so differences between models are paired."""
    def __init__(self, y, n_boot=1000, seed=SEED):
        self.y = np.asarray(y, dtype=np.int8); self.n = len(self.y); self.B = n_boot; self.seed = seed

    def weights(self, b):
        return np.random.default_rng(self.seed + b).poisson(1.0, self.n).astype(np.float32)

    def prepare(self, s):
        order = np.argsort(-s, kind="stable"); ss = s[order]
        ends = np.r_[np.nonzero(ss[1:] != ss[:-1])[0], self.n - 1]           # last index of each tie group
        return order, ends

    def ap(self, prep, w):
        order, ends = prep; wy = w * self.y; wo, wyo = w[order], wy[order]
        ctp, call = np.cumsum(wyo, dtype=np.float64)[ends], np.cumsum(wo, dtype=np.float64)[ends]
        tot = ctp[-1]
        if tot <= 0: return np.nan
        rec = ctp / tot; prec = np.divide(ctp, call, out=np.zeros_like(ctp), where=call > 0)
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))

    def auc(self, prep, w):
        order, ends = prep; wo = w[order]; wp = (w * self.y)[order]; wn = wo - wp
        cp, cn = np.cumsum(wp, dtype=np.float64)[ends], np.cumsum(wn, dtype=np.float64)[ends]
        gp, gn = np.diff(np.r_[0.0, cp]), np.diff(np.r_[0.0, cn]); below = cn[-1] - cn
        return float(np.sum(gp * (below + 0.5 * gn)) / (cp[-1] * cn[-1]))

    def run(self, scores: dict, metrics=("ap",)):
        """-> {metric: {key: array of B bootstrap values}}"""
        preps = {k: self.prepare(np.asarray(v)) for k, v in scores.items()}
        out = {m: {k: np.empty(self.B) for k in scores} for m in metrics}
        for b in range(self.B):
            w = self.weights(b)
            for k in scores:
                for m in metrics: out[m][k][b] = getattr(self, m)(preps[k], w)
        return out


# =====================================================================================================================
# Shared modelling building blocks (single source of truth for 04_train_compare.py AND demo_live.py)
# =====================================================================================================================
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer


def ordinal_codes(df, feats):
    """Feature frame with float32 numerics and ordinal integer codes for categoricals (tree / NB / OHE input)."""
    return pd.DataFrame({c: (df[c].cat.codes.to_numpy().astype("float32") if c in CAT_COLS else df[c].to_numpy("float32")) for c in feats}, index=df.index)


def sk_pre(feats, kind):
    """sklearn preprocessing per model family. Median imputation + missing-indicator columns, fitted on the training rows only."""
    num = [c for c in feats if c in NUM_COLS]; cat = [c for c in feats if c in CAT_COLS]
    imp = SimpleImputer(strategy="median", add_indicator=True)
    if kind == "tree":   return ColumnTransformer([("n", imp, num), ("c", "passthrough", cat)], sparse_threshold=0)
    if kind == "linear": return ColumnTransformer([("n", Pipeline([("i", imp), ("s", StandardScaler())]), num),
                                                   ("c", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200), cat)], sparse_threshold=1.0)
    if kind == "knn":    return ColumnTransformer([("n", Pipeline([("i", imp), ("s", StandardScaler())]), num)], sparse_threshold=0)
    if kind == "gauss":  return ColumnTransformer([("n", imp, num)], sparse_threshold=0)
    if kind == "catnb":  return ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")),
                                   ("b", KBinsDiscretizer(10, encode="ordinal", strategy="quantile", subsample=200_000, random_state=SEED))]), num), ("c", "passthrough", cat)], sparse_threshold=0)
    raise ValueError(kind)


def stratified_idx(y, mask, n, seed):
    """Stratified (by delayed label) random sample of n row positions among rows where mask is True."""
    rng = np.random.default_rng(seed); idx = np.nonzero(mask)[0]
    if n >= len(idx): return idx
    out = []
    for cls in (0, 1):
        ic = idx[y[idx] == cls]; k = int(round(n * len(ic) / len(idx))); out.append(rng.choice(ic, k, replace=False))
    return np.sort(np.concatenate(out))


def sk_model_spec(name, seed, n_jobs=10, dt_depth=12, card=None, feats=None):
    """Hyperparameters of the sklearn baselines as chosen in Step 4 -> dict(kind, model, weighted)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import CategoricalNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    if name == "knn": return dict(kind="knn", weighted=False, model=KNeighborsClassifier(50, algorithm="brute", n_jobs=n_jobs))
    if name == "random_forest": return dict(kind="tree", weighted=True, model=RandomForestClassifier(150, min_samples_leaf=50, max_features="sqrt", n_jobs=n_jobs, random_state=seed))
    if name == "decision_tree": return dict(kind="tree", weighted=True, model=DecisionTreeClassifier(max_depth=dt_depth, min_samples_leaf=50, random_state=seed))
    if name == "logistic_regression": return dict(kind="linear", weighted=True, model=LogisticRegression(C=1.0, max_iter=200, solver="lbfgs"))
    if name == "naive_bayes_gaussian": return dict(kind="gauss", weighted=False, model=GaussianNB(priors=[0.5, 0.5]))
    if name == "naive_bayes_categorical":
        nn = [c for c in feats if c in NUM_COLS]; ncat = [c for c in feats if c in CAT_COLS]
        return dict(kind="catnb", weighted=False, model=CategoricalNB(min_categories=[10] * len(nn) + [card[c] + 1 for c in ncat], class_prior=[0.5, 0.5]))
    raise ValueError(name)


LGB_BASE = dict(objective="binary", metric="average_precision", bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20,
                cat_l2=10, min_data_per_group=100, verbose=-1, max_bin=255)
XGB_DEFAULT = dict(learning_rate=0.1, max_depth=8, min_child_weight=5, colsample_bytree=0.8)


def make_xgb(params, seed, native=False, n_jobs=10, early=True):
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=2000, subsample=0.8, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50 if early else None,
                             n_jobs=n_jobs, random_state=seed, enable_categorical=native, max_cat_to_onehot=4, **params)
