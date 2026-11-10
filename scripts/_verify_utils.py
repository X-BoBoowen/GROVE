"""Small shared numerical utilities; run verify_*.py with the prescribed uv command."""
import itertools
import numpy as np


def laplacian(n, edges):
    a = np.zeros((n, n))
    for u, v, w in edges:
        a[u, v] += w
        a[v, u] += w
    return np.diag(a.sum(axis=1)) - a


def subsets(n):
    return [frozenset(i for i in range(n) if mask >> i & 1)
            for mask in range(1 << n)]


def grounded(a, selected, beta):
    if beta == np.inf:
        keep = [i for i in range(len(a)) if i not in selected]
        return a[np.ix_(keep, keep)]
    return a + np.diag([beta if i in selected else 0.0 for i in range(len(a))])


def trace_function(a, f):
    return float(np.sum(f(np.linalg.eigvalsh(a)))) if len(a) else 0.0


def check_set_function(values, n, tol):
    worst_mono = worst_mixed = 0.0
    for s, value in values.items():
        rest = [i for i in range(n) if i not in s]
        for v in rest:
            drop = value - values[s | {v}]
            scale = max(1.0, abs(value), abs(values[s | {v}]))
            worst_mono = min(worst_mono, drop / scale)
            assert drop >= -tol * scale, ("monotonicity", s, v, drop)
        for u, v in itertools.combinations(rest, 2):
            terms = [value, values[s | {u}], values[s | {v}], values[s | {u, v}]]
            mixed = terms[0] - terms[1] - terms[2] + terms[3]
            scale = max(1.0, *map(abs, terms))
            worst_mixed = min(worst_mixed, mixed / scale)
            assert mixed >= -tol * scale, ("supermodularity", s, u, v, mixed)
    return worst_mono, worst_mixed


def run(name, test):
    try:
        test()
    except Exception as exc:
        print(f"FAIL {name}: {exc}", flush=True)
        raise


def scalar_resolvent_drop(a, v, beta, order=1, shift=0.0):
    """Recover the order-th trace drop using scalar resolvent moments only."""
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    weights = eigenvectors[v]**2
    moments = [np.sum(weights/(eigenvalues+shift)**j) for j in range(1, order+2)]
    # Taylor coefficients of F(z)=1+beta*r1(t+z), or F(z)=r1(t+z).
    f = np.array([(-1)**j*moments[j] for j in range(order+1)])
    if beta != np.inf:
        f *= beta
        f[0] += 1
    g = np.zeros(order+1)
    for k in range(1, order+1):
        g[k] = (k*f[k]-sum(j*g[j]*f[k-j] for j in range(1, k)))/(k*f[0])
    return (-1)**order*order*g[order]


def updated(a, v, beta):
    return grounded(a, {v}, beta)


def fixtures():
    l = laplacian(6, [(0, 1, .7), (1, 2, 1.3), (2, 3, .4),
                      (3, 4, 1.1), (4, 5, .9), (0, 5, .8), (1, 4, .6)])
    return [l+.2*np.eye(6), grounded(l, {0}, 1.2),
            grounded(l, {0}, np.inf), np.array([[1.7]])]

