"""Pilot 2: large exponents s in {2,3,4,8} as supermodular surrogates for the (non-submodular) grounded
lambda_min objective. Hard grounding of M=L+sigma I. Exact marginals for integer m via the log-series
recurrence of theory.tex Prop. allpowers at t=0: f_j=(-1)^j r_{j+1}, r_j=e_v^T X^{-j} e_v; Delta_m=q_m(0)=(-1)^m m g_m.
Baselines: eigvec = pick argmax |phi_min(v)| each round (first-order heuristic for lambda_min);
lmin_exact = exact greedy on lambda_min (only for n<=1200)."""
import sys, json, time, argparse, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from s_sensitivity import load_graph, downstream
import scipy.sparse as sp

def delta_int(lam, U2, m):
    r = [None] + [U2 @ lam ** (-j) for j in range(1, m + 2)]       # r[j], j=1..m+1
    f = [((-1) ** j) * r[j + 1] for j in range(m + 1)]              # hard: f_j = (-1)^j r_{j+1}
    g = [None] * (m + 1)
    for k in range(1, m + 1):
        acc = k * f[k]
        for j in range(1, k): acc = acc - j * g[j] * f[k - j]
        g[k] = acc / (k * f[0])
    return ((-1) ** m) * m * g[m]

def greedy(M, k, plan):
    n = M.shape[0]; alive = np.arange(n); S = []
    for r in range(k):
        X = M[np.ix_(alive, alive)]; lam, U = np.linalg.eigh(X)
        if plan == "eigvec": d = np.abs(U[:, 0])
        elif plan == "lmin_exact":
            d = np.array([np.linalg.eigvalsh(np.delete(np.delete(X, j, 0), j, 1))[0] for j in range(len(alive))])
        else: d = delta_int(lam, U * U, int(plan[1:]))
        j = int(np.argmax(d)); S.append(int(alive[j])); alive = np.delete(alive, j)
    return S

ap = argparse.ArgumentParser(); ap.add_argument("graph"); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=20); ap.add_argument("--sigma", type=float, default=1e-2); a = ap.parse_args()
A = load_graph(a.graph); n = A.shape[0]
M = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).toarray() + a.sigma * np.eye(n)
# self-check of the recurrence against exact deletion at m=1,2,3
lam, U = np.linalg.eigh(M[:60, :60] + np.eye(60))
for m in (1, 2, 3):
    X = M[:60, :60] + np.eye(60); d = delta_int(lam, U * U, m)[7]
    Y = np.delete(np.delete(X, 7, 0), 7, 1); ex = (np.linalg.eigvalsh(X) ** -m).sum() - (np.linalg.eigvalsh(Y) ** -m).sum()
    print(f"check m={m} rel err {abs(d-ex)/ex:.2e}", flush=True)
plans = ["s1", "s2", "s3", "s4", "s8", "eigvec"] + (["lmin_exact"] if n <= 1200 else [])
res = {"graph": a.graph, "n": n, "k": a.k, "sets": {}, "metrics": {}}
for p in plans:
    t0 = time.time(); res["sets"][p] = greedy(M, a.k, p); print(f"{p}: {time.time()-t0:.1f}s", flush=True)
    res["metrics"][p] = downstream(M, res["sets"][p], [1, 10, 100])
json.dump(res, open(a.out, "w"), indent=1); print("done")
