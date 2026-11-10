"""Pilot (go/no-go for paper 9): does the exponent s change the greedy grounded group on real graphs,
and does that change downstream leader-follower behaviour?

Exact dense computation. Hard grounding of M = L + sigma*I. Greedy with exact marginals:
  s in (0,1): Delta_v = c_s * int_0^inf t^{-s} r2/r1 dt (Gauss-Jacobi, q nodes; spot-checked vs eigendecomposition)
  s = 1:      Delta_v = (A^2)_vv / A_vv,  A = X^{-1}
  s = 2:      Delta_v = 2 (A^3)_vv / A_vv - ((A^2)_vv / A_vv)^2
  logdet (s->0 limit): Delta(tr X^{-s}) = 1 + s*log A_vv + o(s), so rank by log A_vv
plus degree baseline. Downstream metrics on each final set: grounded heat trace tr exp(-tau X_S) for several tau,
lambda_min(X_S), tr(X_S^{-1}) (CFCC denominator).
"""
import argparse, json, math, os, sys, time, gzip
import numpy as np
from scipy.special import roots_jacobi
import scipy.sparse as sp
import scipy.sparse.csgraph as csg


def load_graph(path):
    op = gzip.open if path.endswith(".gz") else open
    E = []
    with op(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line[0] in "#%":
                continue
            a, b = line.split()[:2]
            if a != b:
                E.append((a, b))
    ids = {}
    for a, b in E:
        ids.setdefault(a, len(ids)); ids.setdefault(b, len(ids))
    n = len(ids)
    r = np.array([ids[a] for a, _ in E]); c = np.array([ids[b] for _, b in E])
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    ncomp, lab = csg.connected_components(A, directed=False)
    big = np.argmax(np.bincount(lab))
    keep = np.where(lab == big)[0]
    A = A[keep][:, keep]
    return A


def marginals(lam, U2, kind, s=None, q=64):
    """Delta_v for all v of current X = U diag(lam) U^T; U2 = U**2 (rows = vertices)."""
    if kind == "s1":
        a1 = U2 @ (1 / lam); a2 = U2 @ (1 / lam**2); return a2 / a1
    if kind == "s2":
        a1 = U2 @ (1 / lam); a2 = U2 @ (1 / lam**2); a3 = U2 @ (1 / lam**3)
        return 2 * a3 / a1 - (a2 / a1) ** 2
    if kind == "logdet":
        return np.log(U2 @ (1 / lam))   # s->0: Delta(tr X^{-s}) ~ 1 + s*log A_vv
    if kind == "frac":
        cs = math.sin(math.pi * s) / math.pi
        x, w = roots_jacobi(q, s - 1.0, -s)          # weight (1-x)^{s-1}(1+x)^{-s}
        c = math.sqrt(lam.min() * lam.max())
        t = c * (1 + x) / (1 - x)
        # t^{-s} dt = 2 c^{1-s} (1+x)^{-s} (1-x)^{s-2} dx  ->  integrand g(t) * 2c^{1-s}/(1-x)
        inv = 1.0 / (lam[:, None] + t[None, :])        # (d, q)
        r1 = U2 @ inv; r2 = U2 @ inv**2
        g = r2 / r1
        return cs * (g * (2 * c ** (1 - s) * w / (1 - x))[None, :]).sum(1)
    raise ValueError(kind)


def exact_obj(X, kind, s=None):
    lam = np.linalg.eigvalsh(X)
    if kind == "frac": return (lam ** -s).sum()
    if kind == "s1": return (1 / lam).sum()
    if kind == "s2": return (lam ** -2.0).sum()
    if kind == "logdet": return -np.log(lam).sum()


def greedy(M, k, kind, s=None, check=False, log=None):
    n = M.shape[0]; S = []; alive = np.arange(n)
    for r in range(k):
        X = M[np.ix_(alive, alive)]
        lam, U = np.linalg.eigh(X)
        d = marginals(lam, U * U, kind, s)
        j = int(np.argmax(d))
        if check and r == 0:  # spot-check quadrature / closed form against exact deletion
            for jj in [j, int(np.argsort(d)[len(d) // 2])]:
                keep = np.delete(np.arange(len(alive)), jj)
                ex = exact_obj(X, kind, s) - exact_obj(X[np.ix_(keep, keep)], kind, s)
                if kind == "logdet": ex -= 0.0
                rel = abs(ex - d[jj]) / abs(ex) if kind != "logdet" else abs(ex - d[jj])
                if log: log(f"    check {kind} s={s}: formula={d[jj]:.10g} exact={ex:.10g} err={rel:.2e}")
        S.append(int(alive[j])); alive = np.delete(alive, j)
    return S


def downstream(M, S, taus):
    keep = np.setdiff1d(np.arange(M.shape[0]), S)
    lam = np.linalg.eigvalsh(M[np.ix_(keep, keep)])
    out = {f"heat_tau={t:g}": float(np.exp(-t * lam).sum()) for t in taus}
    out["lambda_min"] = float(lam[0]); out["tr_inv"] = float((1 / lam).sum())
    for s in (0.25, 0.5): out[f"tr_pow_-{s}"] = float((lam ** -s).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph"); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=1e-2); ap.add_argument("--out", required=True)
    ap.add_argument("--max-n", type=int, default=12000)
    a = ap.parse_args()
    logf = open(a.out + ".log", "a")
    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()
    A = load_graph(a.graph); n = A.shape[0]
    if n > a.max_n: log(f"skip {a.graph}: n={n}"); return
    L = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).toarray()
    M = L + a.sigma * np.eye(n)
    log(f"graph {os.path.basename(a.graph)} n={n} m={int(A.nnz // 2)} sigma={a.sigma} k={a.k}")
    plans = [("logdet", None), ("frac", 0.25), ("frac", 0.5), ("frac", 0.75), ("s1", None), ("s2", None)]
    res = {"graph": a.graph, "n": n, "m": int(A.nnz // 2), "sigma": a.sigma, "k": a.k, "sets": {}, "metrics": {}}
    deg = np.asarray(A.sum(1)).ravel()
    res["sets"]["degree"] = [int(v) for v in np.argsort(-deg, kind="stable")[: a.k]]
    for kind, s in plans:
        name = kind if s is None else f"s={s}"
        t0 = time.time(); S = greedy(M, a.k, kind, s, check=True, log=log)
        res["sets"][name] = S; log(f"  {name}: {time.time() - t0:.1f}s  S[:8]={S[:8]}")
    taus = [0.01, 0.1, 1, 10, 100]
    for name, S in res["sets"].items():
        res["metrics"][name] = downstream(M, S, taus)
    base = set(res["sets"]["s1"])
    for name, S in res["sets"].items():
        ov = len(base & set(S)); log(f"  overlap with CFCC(s=1) {name}: {ov}/{a.k}")
    json.dump(res, open(a.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
