"""Validate the C++ engine against dense numpy on a small graph.
usage: python validate.py <ground binary> <graph.bin>
Checks: seed (argmin L^+_vv), exact refresh marginals p=1..4,8 vs dense deletion, dense greedy vs celf/jlg sets,
eval (tr X^-1, tr X^-2, lambda_min) vs dense."""
import sys, json, subprocess, tempfile, numpy as np

exe, gpath = sys.argv[1], sys.argv[2]


def load(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GRND"
        n = int(np.frombuffer(f.read(4), np.int32)[0]); nnz = int(np.frombuffer(f.read(8), np.int64)[0])
        ptr = np.frombuffer(f.read(8 * (n + 1)), np.int64); adj = np.frombuffer(f.read(4 * nnz), np.int32)
    L = np.zeros((n, n))
    for i in range(n):
        L[i, adj[ptr[i]:ptr[i + 1]]] = -1
        L[i, i] = ptr[i + 1] - ptr[i]
    return L


def run(*args):
    out = subprocess.run([exe, *map(str, args)], capture_output=True, text=True, check=True).stdout
    return out


L = load(gpath); n = L.shape[0]
Lp = np.linalg.pinv(L)
seed_exact = int(np.argmin(np.diag(Lp)))
js = json.loads(run("seed", gpath, "--d", 512))
d = np.diag(Lp)
print(f"seed: engine {js['seed']} exact {seed_exact}  L+ engine/exact {d[js['seed']]:.6g}/{d[seed_exact]:.6g}")
seed = seed_exact

keep = np.setdiff1d(np.arange(n), [seed])
X = L[np.ix_(keep, keep)]
lam, U = np.linalg.eigh(X)
U2 = U * U
tr = lambda p, lm: (lm ** -float(p)).sum()
ok = True
for p in (1, 2, 3, 4, 8):
    lines = run("exactmarg", gpath, "--p", p, "--seed", seed, "--rb", 32, "--rtol", 1e-11).split("\n")
    eng = {int(l.split()[0]): float(l.split()[1]) for l in lines if l.strip()}
    rng = np.random.default_rng(p)
    errs = []
    for jj in rng.choice(len(keep), 25, replace=False):
        v = keep[jj]
        Y = np.delete(np.delete(X, jj, 0), jj, 1)
        ex = tr(p, lam) - tr(p, np.linalg.eigvalsh(Y))
        errs.append(abs(eng[v] - ex) / abs(ex))
    print(f"exact refresh p={p}: max rel err vs dense deletion {max(errs):.2e}")
    ok &= max(errs) < 1e-5

for p in (1, 2, 4):
    dense = json.loads(run("dense", gpath, "--p", p, "--k", 10, "--seed", seed))["set"]
    celf = json.loads(run("celf", gpath, "--p", p, "--k", 10, "--seed", seed, "--d", 128))
    jlg = json.loads(run("jlg", gpath, "--p", p, "--k", 10, "--seed", seed, "--d", 128))["set"]
    print(f"p={p}: dense {dense}\n      celf  {celf['set']} refreshed={celf['refreshed']} {celf['round_refresh']}\n"
          f"      jlg   {jlg}  overlap(celf,dense)={len(set(dense)&set(celf['set']))} overlap(jlg,dense)={len(set(dense)&set(jlg))}")
    lazy = json.loads(run("celf", gpath, "--p", p, "--k", 10, "--seed", seed, "--slack", 1e9, "--rebuild", 0, "--rtol", 1e-10))["set"]
    print(f"      exact-lazy CELF == dense greedy: {lazy == dense}")
    ok &= lazy == dense
    S = celf["set"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as sf:
        sf.write(" ".join(map(str, S))); sf.flush()
        ev = json.loads(run("eval", gpath, "--set", sf.name, "--probes", 512))
    kk = np.setdiff1d(np.arange(n), S); lm = np.linalg.eigvalsh(L[np.ix_(kk, kk)])
    print(f"      eval tr_inv {ev['tr_inv']:.6g}±{ev['tr_inv_se']:.2g} exact {tr(1, lm):.6g}; tr_inv2 {ev['tr_inv2']:.6g} exact {tr(2, lm):.6g};"
          f" lambda_min {ev['lambda_min']:.8g} exact {lm[0]:.8g} (steps {ev['lanczos_steps']})")
    ok &= abs(ev["lambda_min"] / lm[0] - 1) < 1e-6
print("ALL PASS" if ok else "SOME CHECK FAILED")
