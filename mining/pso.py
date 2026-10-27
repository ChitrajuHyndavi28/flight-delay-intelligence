"""Minimal particle swarm optimiser (the tuner used by the base paper) over a mixed log / linear / integer search space."""
import json, math
import numpy as np


def decode(u, space):
    out = {}
    for x, (name, lo, hi, kind) in zip(u, space):
        v = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))) if kind == "log" else lo + x * (hi - lo)
        out[name] = int(round(v)) if kind == "int" else (int(round(math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))))) if kind == "logint" else float(v))
    return out


def pso(objective, space, n_particles=8, n_iter=5, seed=42, w=0.7, c1=1.5, c2=1.5, log_path=None):
    """Maximise objective(params dict). Returns (best_params, best_score, history)."""
    rng = np.random.default_rng(seed); d = len(space)
    pos = rng.random((n_particles, d)); vel = rng.normal(0, 0.1, (n_particles, d))
    pbest, pscore = pos.copy(), np.full(n_particles, -np.inf); gbest, gscore = pos[0].copy(), -np.inf; hist = []
    for it in range(n_iter + 1):
        for i in range(n_particles):
            p = decode(pos[i], space); s = float(objective(p)); hist.append(dict(iter=it, particle=i, score=s, **p))
            print(f"  PSO it {it} p{i}: {s:.4f} {p}", flush=True)
            if s > pscore[i]: pscore[i], pbest[i] = s, pos[i].copy()
            if s > gscore: gscore, gbest = s, pos[i].copy()
            if log_path: json.dump(hist, open(log_path, "w"))
        r1, r2 = rng.random((n_particles, d)), rng.random((n_particles, d))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos); pos = np.clip(pos + vel, 0, 1)
    return decode(gbest, space), gscore, hist
