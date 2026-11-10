"""Per-round moment/key error of the scalar cache vs. maintained solution sketches (ca-HepPh).

Dense exact emulation, d = 64, K = 20, common exact-greedy trajectory from the exact argmin L^+_vv seed.
  scalar cache : JL moments estimated once at S0, then exact decrements (eqs. mainr1/mainr2 of the paper).
  sketches     : Y1 = Pi_C C R, Y2 = Pi_I R updated by the rank-one identity (eq. mainbankupdate);
                 these are the keys GROVE uses for unrefreshed candidates (refreshed ones are exact).
Checks (asserted):
  (a) LCC size equals the paper's (11204, 117619);
  (b) exact greedy gains equal runs/s2/med/ca-HepPh.p1.json (engine, exact init) to 1e-6 relative;
  (c) the scalar cache keeps its signed absolute error (Proposition "Hybrid state identities");
  (d) the maintained sketches equal fresh sketches Pi_C C R_T, Pi_I R_T with the same randomness.
Usage: python -I scripts/fig_error_trajectory.py <ca-HepPh.txt> [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

D, K, NPROJ, TOPQ = 64, 20, 5, 100
ROOT = Path(__file__).resolve().parent.parent


def load_lcc(path):
    e = np.loadtxt(path, dtype=np.int64, comments="#")
    e = e[e[:, 0] != e[:, 1]]
    ids, inv = np.unique(e, return_inverse=True)
    e = inv.reshape(-1, 2)
    n = len(ids)
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64)
    _, lab = connected_components(a, directed=False)
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    return a[keep][:, keep].tocsr()


def main():
    a = load_lcc(sys.argv[1])
    n, m = a.shape[0], a.nnz // 2
    assert (n, m) == (11204, 117619), (n, m)
    deg = np.asarray(a.sum(axis=1)).ravel()
    lap = np.diag(deg) - a.toarray()
    ref = json.loads((ROOT / "runs/s2/med/ca-HepPh.p1.json").read_text())["gains"]
    lp = np.linalg.inv(lap + 1.0 / n) - 1.0 / n
    lpinv_diag = np.diag(lp).copy()
    del lap
    # The engine's seed is a JL estimate of argmin L^+_vv; identify it among the 20 smallest exact
    # diagonals by its first exact greedy gain (grounded inverse: R_uv = L+_uv - L+_us - L+_sv + L+_ss).
    N, s0, R = n - 1, None, None
    for rank, s in enumerate(np.argsort(lpinv_diag)[:20]):
        w = np.delete(np.arange(n), s)
        Rs = lp[np.ix_(w, w)] - lp[w, s][:, None] - lp[s, w][None, :] + lp[s, s]
        g1 = ((Rs ** 2).sum(0) / np.diag(Rs)).max()
        if abs(g1 - ref[0]) < 1e-6 * ref[0]:
            s0, R, seed_rank = int(s), Rs, rank
            break
    assert s0 is not None, "engine seed not among the 20 smallest L^+_vv"
    print(f"seed: exact L^+_vv rank {seed_rank} (0 = argmin), value {lpinv_diag[s0]:.6f} vs min {lpinv_diag.min():.6f}")
    w = np.delete(np.arange(n), s0)
    del lp

    # oriented incidence restricted to W (m x N)
    up = sp.triu(a, k=1).tocoo()
    pos = np.full(n, -1)
    pos[w] = np.arange(N)
    rows = np.concatenate([np.arange(m), np.arange(m)])
    cols = np.concatenate([pos[up.row], pos[up.col]])
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    ok = cols >= 0
    C = sp.coo_matrix((vals[ok], (rows[ok], cols[ok])), shape=(m, N)).tocsr()

    rng = np.random.default_rng(20261110)
    PC, PI, Y1, Y2, c1, c2 = [], [], [], [], [], []
    for _ in range(NPROJ):
        pc = (C.T @ (rng.choice([-1.0, 1.0], size=(m, D)) / np.sqrt(D))).T  # Pi_C C, d x N
        pi = rng.choice([-1.0, 1.0], size=(D, N)) / np.sqrt(D)
        PC.append(pc); PI.append(pi)
        Y1.append(pc @ R); Y2.append(pi @ R)
        c1.append((Y1[-1] ** 2).sum(0)); c2.append((Y2[-1] ** 2).sum(0))

    active = np.ones(N, bool)
    r1, r2 = np.diag(R).copy(), (R ** 2).sum(0)
    err0 = [(c1[j] - r1, c2[j] - r2) for j in range(NPROJ)]
    names = ["scalar", "sketch"]
    out = {"n": n, "m": m, "d": D, "K": K, "projections": NPROJ, "seed_lpinv": float(lpinv_diag[s0]),
           "gains": [], "rounds": []}
    worst_c, worst_d = 0.0, 0.0
    for t in range(K - 1):
        r1, r2 = np.diag(R).copy(), (R ** 2).sum(0)
        idx = np.flatnonzero(active)
        key = r2[idx] / r1[idx]
        top = np.argsort(-key)[:TOPQ]
        rec = {"round": t + 1}
        for name in names:
            acc = {q: [] for q in ("key_med", "key_p90", "key_p99", "key_max", "key_top_med", "r2_med", "r1_med", "regret")}
            for j in range(NPROJ):
                if name == "scalar":
                    e1, e2 = c1[j][idx], c2[j][idx]
                    worst_c = max(worst_c, np.abs(e1 - r1[idx] - err0[j][0][idx]).max() / r1[idx].max(),
                                  np.abs(e2 - r2[idx] - err0[j][1][idx]).max() / r2[idx].max())
                else:
                    e1, e2 = (Y1[j][:, idx] ** 2).sum(0), (Y2[j][:, idx] ** 2).sum(0)
                ek = e2 / e1
                rel = np.abs(ek - key) / key
                acc["key_med"].append(np.median(rel)); acc["key_p90"].append(np.quantile(rel, 0.9))
                acc["key_p99"].append(np.quantile(rel, 0.99)); acc["key_max"].append(rel.max())
                acc["key_top_med"].append(np.median(rel[top]))
                acc["r2_med"].append(np.median(np.abs(e2 - r2[idx]) / r2[idx]))
                acc["r1_med"].append(np.median(np.abs(e1 - r1[idx]) / r1[idx]))
                acc["regret"].append(1.0 - key[np.argmax(ek)] / key.max())
            rec[name] = {q: float(np.mean(v)) for q, v in acc.items()}
        out["rounds"].append(rec)
        v = idx[np.argmax(key)]
        out["gains"].append(float(key.max()))
        x = R[:, v].copy(); y = R @ x; c = 1.0 / x[v]; sx = x @ x
        for j in range(NPROJ):
            c1[j] = c1[j] - c * x * x
            c2[j] = c2[j] - 2 * c * x * y + c * c * sx * x * x
            Y1[j] -= c * np.outer(Y1[j][:, v], x)
            Y2[j] -= c * np.outer(Y2[j][:, v], x)
        for lo in range(0, N, 1024):
            R[lo:lo + 1024] -= c * np.outer(x[lo:lo + 1024], x)
        R[v, :] = 0.0; R[:, v] = 0.0
        active[v] = False
        print(f"round {t+1:2d} gain {out['gains'][-1]:.7f}  "
              f"scalar key med {rec['scalar']['key_med']:.3f} p99 {rec['scalar']['key_p99']:.3f} regret {rec['scalar']['regret']:.3f} | "
              f"sketch key med {rec['sketch']['key_med']:.3f} p99 {rec['sketch']['key_p99']:.3f} regret {rec['sketch']['regret']:.3f}",
              flush=True)

    for j in range(NPROJ):  # (d) maintained == fresh
        for Y, P in ((Y1[j], PC[j]), (Y2[j], PI[j])):
            worst_d = max(worst_d, np.abs(Y - P @ R).max() / np.abs(Y).max())
    gain_dev = float(np.max(np.abs(np.array(out["gains"]) - np.array(ref)) / np.array(ref)))
    out["checks"] = {"gain_dev_vs_engine": gain_dev, "scalar_error_drift": float(worst_c),
                     "sketch_vs_fresh": float(worst_d)}
    print("checks", out["checks"])
    assert gain_dev < 1e-6 and worst_c < 1e-8 and worst_d < 1e-8
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs/s3/hepph_error_trajectory.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    for name in names:
        for q in ("key_med", "key_p90", "key_p99", "key_max", "regret"):
            print(name, q, " ".join(f"({r['round']},{100*r[name][q]:.2f})" for r in out["rounds"]))
    print("PASS")


if __name__ == "__main__":
    main()
