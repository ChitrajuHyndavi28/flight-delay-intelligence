"""Step 4c: neural network (embedding MLP, PyTorch) + score-averaging ensemble, same split / threshold protocol as Step 4.
Writes cached predictions to mining_cache/preds/<scenario>/{mlp,ensemble}__<seed>.* and results/model_comparison/nn_and_ensemble_*.csv."""
import argparse, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from common import ROOT, SEED
from ml_common import (CACHE, CAT_COLS, NUM_COLS, SCENARIOS, TARGET, assert_no_leakage, best_f1_threshold, class_weights,
                       load_features, metrics_at, split_masks)

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", default=["t60", "dayahead"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42])
ap.add_argument("--epochs", type=int, default=14)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--tune", action="store_true", help="PSO-tune the MLP on a train sample (validation PR-AUC), then train the final model with the best values")
args = ap.parse_args()

PRED = CACHE / "preds"; OUT = ROOT / "results" / "model_comparison"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = load_features()
if args.smoke:
    df = pd.concat([d.sample(min(len(d), n), random_state=0) for d, n in
                    [(df[df.split == "train"], 60_000), (df[df.split == "val"], 20_000), (df[df.split == "test"], 20_000)]])
tr, va, te = split_masks(df)
y = df[TARGET].to_numpy(np.float32)
assert_no_leakage(SCENARIOS["t60"])


def prep(feats):
    """Numerics: train-median impute + missing indicator + standardise (fit on train). Categoricals: integer ids, rare -> 0."""
    num = [c for c in feats if c in NUM_COLS]; cat = [c for c in feats if c in CAT_COLS]
    Xn = df[num].to_numpy("float32")
    med = np.nanmedian(Xn[tr], axis=0); miss = np.isnan(Xn)
    keep = miss[tr].any(axis=0)                                             # indicator only where train has NaN
    Xn = np.where(miss, med, Xn)
    mu, sd = Xn[tr].mean(0), Xn[tr].std(0) + 1e-6
    Xn = np.clip((Xn - mu) / sd, -6, 6)
    Xn = np.hstack([Xn, miss[:, keep].astype("float32")]).astype("float32")
    Xc, card = [], []
    for c in cat:
        codes = df[c].cat.codes.to_numpy().astype("int64") + 1               # NaN -> 0
        cnt = np.bincount(codes[tr], minlength=len(df[c].cat.categories) + 1)
        codes = np.where(cnt[codes] < 20, 0, codes)                            # rare / unseen-in-train -> 0
        Xc.append(codes); card.append(len(df[c].cat.categories) + 1)
    return torch.from_numpy(Xn), torch.from_numpy(np.stack(Xc, 1)), card


class Net(nn.Module):
    def __init__(self, n_num, card, hidden=(512, 256, 128), drop=0.25):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(k, min(24, int(np.ceil(k ** 0.5)) + 1)) for k in card])
        d = n_num + sum(e.embedding_dim for e in self.emb); layers = []
        for h in hidden: layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]; d = h
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, xn, xc):
        return self.mlp(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], 1)).squeeze(1)


@torch.no_grad()
def predict(model, Xn, Xc, idx, bs=65536):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]; out.append(torch.sigmoid(model(Xn[j].to(DEV), Xc[j].to(DEV))).cpu().numpy())
    return np.concatenate(out)


def train_mlp(scn, seed, hp=None, sample=None, epochs=None):
    hp = hp or dict(width=512, drop=0.25, lr=3e-3, wd=1e-4, bs=4096); epochs = epochs or args.epochs
    torch.manual_seed(seed); np.random.seed(seed)
    feats = SCENARIOS[scn]; Xn, Xc, card = prep(feats)
    itr, iva, ite = [np.nonzero(m)[0] for m in (tr, va, te)]
    if sample: itr = np.sort(np.random.default_rng(seed).choice(itr, sample, replace=False))
    cw = class_weights(y[tr]); model = Net(Xn.shape[1], card, hidden=(hp["width"], hp["width"] // 2, hp["width"] // 4), drop=hp["drop"]).to(DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cw[1] / cw[0], device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"]); bs = int(hp["bs"])
    steps = epochs * int(np.ceil(len(itr) / bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps)
    best, best_state, bad, t0 = -1, None, 0, time.time(); yt = torch.from_numpy(y)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(itr); tot = 0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            if len(j) < 2: continue
            opt.zero_grad(); l = lossf(model(Xn[j].to(DEV), Xc[j].to(DEV)), yt[j].to(DEV)); l.backward(); opt.step(); sched.step(); tot += l.item() * len(j)
        pv = predict(model, Xn, Xc, iva); pr = average_precision_score(y[va], pv)
        print(f"[{scn} s{seed}] epoch {ep + 1}/{epochs} loss {tot / len(perm):.4f} val PR-AUC {pr:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if pr > best: best, bad, best_state = pr, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    model.load_state_dict(best_state); tt = time.time() - t0; t1 = time.time()
    pv, pt = predict(model, Xn, Xc, iva), predict(model, Xn, Xc, ite)
    return pv, pt, tt, time.time() - t1, int(len(itr))


def save(scn, name, seed, val, test, tt, pt, n, note):
    d = PRED / scn; d.mkdir(exist_ok=True, parents=True)
    np.savez(d / f"{name}__{seed}.npz", val=np.asarray(val, "float32"), test=np.asarray(test, "float32"))
    json.dump(dict(model=name, scenario=scn, seed=seed, train_time_s=round(tt, 2), predict_time_s=round(pt, 2), n_train=n, note=note),
              open(d / f"{name}__{seed}.json", "w"))


def f2(p, r): return 5 * p * r / (4 * p + r) if p + r > 0 else 0.0


def evaluate(label, val, test, yv, yt):
    thr, _ = best_f1_threshold(yv, val); m = metrics_at(yt, test, thr)
    # paper-style operating point: lowest-threshold-on-validation that gives >= 68% recall (undersampling-like behaviour)
    order = np.sort(val[yv == 1]); thr68 = float(order[int(0.32 * len(order))]); m68 = metrics_at(yt, test, thr68)
    return dict(model=label, pr_auc=m["pr_auc"], roc_auc=m["roc_auc"], accuracy=m["accuracy"], precision=m["precision"], recall=m["recall"],
                f1=m["f1"], f2=f2(m["precision"], m["recall"]),
                acc_at_recall68=m68["accuracy"], prec_at_recall68=m68["precision"], rec_at_recall68=m68["recall"], f2_at_recall68=f2(m68["precision"], m68["recall"]))


def rank01(s): return rankdata(s) / len(s)


rows = {}
for scn in args.scenarios:
    yv, yt = y[va].astype(int), y[te].astype(int); res = []
    hp = None
    if args.tune and not args.smoke:
        from pso import pso
        hpf = OUT / "mlp_pso_best.json"; allbest = json.load(open(hpf)) if hpf.exists() else {}
        if scn not in allbest:
            space = [("width", 128, 1024, "logint"), ("drop", 0.0, 0.5, "lin"), ("lr", 5e-4, 8e-3, "log"), ("wd", 1e-6, 1e-2, "log"), ("bs", 1024, 16384, "logint")]
            def obj(p):
                pv, _, *_ = train_mlp(scn, SEED, p, sample=600_000, epochs=6); return average_precision_score(y[va], pv)
            bp, bs_, _ = pso(obj, space, n_particles=8, n_iter=4, seed=SEED, log_path=OUT / f"mlp_pso_history_{scn}.json")
            allbest[scn] = dict(params=bp, val_pr_auc_sample=bs_); json.dump(allbest, open(hpf, "w"), indent=1)
        hp = allbest[scn]["params"]; print("MLP tuned params", scn, hp, flush=True)
    for seed in args.seeds:
        f = PRED / scn / f"{'mlp_tuned' if hp else 'mlp'}__{seed}.npz"
        if f.exists() and not args.smoke: z = np.load(f); pv, pt = z["val"], z["test"]
        else:
            pv, pt, tt, pts, n = train_mlp(scn, seed, hp)
            if not args.smoke: save(scn, "mlp_tuned" if hp else "mlp", seed, pv, pt, tt, pts, n, "embedding MLP 512-256-128, BN+dropout, class-weighted BCE, OneCycle AdamW")
        res.append(evaluate(f"Neural Network (MLP) seed {seed}", pv, pt, yv, yt))
    # ensemble of the cached top models (seed 42) + the MLP (seed 42): rank-average of probabilities
    members = {"LightGBM": "lightgbm", "XGBoost (native, tuned)": "xgboost_native_tuned"}
    S = {}
    for lab, nm in members.items():
        z = np.load(PRED / scn / f"{nm}__{SEED}.npz"); S[lab] = (z["val"], z["test"]);
    z = np.load(PRED / scn / f"{'mlp_tuned' if hp else 'mlp'}__{args.seeds[0]}.npz") if not args.smoke else None
    if z is not None:
        S["MLP"] = (z["val"], z["test"])
        # val ranks are computed within val, test ranks within test: monotone per-set, so thresholds are tuned on val ranks and applied to test ranks
        ev = np.mean([rank01(v) for v, _ in S.values()], 0); et = np.mean([rank01(t) for _, t in S.values()], 0)
        save(scn, "ensemble_lgb_xgb_mlp_tuned" if hp else "ensemble_lgb_xgb_mlp", SEED, ev, et, 0, 0, int(tr.sum()), "mean of rank-normalised scores")
        res.append(evaluate("Ensemble (LightGBM + XGBoost + MLP)", ev, et, yv, yt))
        for lab, (v, t) in S.items():
            if lab != "MLP": res.append(evaluate(lab + " (reference)", v, t, yv, yt))
    out = pd.DataFrame(res); out.to_csv(OUT / f"nn_and_ensemble_{scn}{'_smoke' if args.smoke else ''}{'_tuned' if hp else ''}.csv", index=False)
    print(f"\n=== {scn} ==="); print(out.round(4).to_string(index=False))
