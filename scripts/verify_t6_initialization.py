"""Two fixed Gaussian RHS families give r1/r2, with deterministic solve-error checks."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase2_utils.py").read_text(), "_phase2_utils.py", "exec"))
import math


def sddm_factor(a):
    rows = []
    n = len(a)
    for u in range(n):
        for v in range(u+1, n):
            if a[u, v] < 0:
                row = np.zeros(n)
                row[u], row[v] = np.sqrt(-a[u, v]), -np.sqrt(-a[u, v])
                rows.append(row)
    for u, value in enumerate(a.sum(axis=1)):
        assert value >= -1e-12
        if value > 1e-12:
            row = np.zeros(n)
            row[u] = np.sqrt(value)
            rows.append(row)
    return np.array(rows)


def test():
    rng = np.random.default_rng(602)
    epsilon, failure = .4, .1
    shifts, max_error, batches = [0, .01, 1, 100], 0.0, 0
    for a in fixtures()[:3]:
        n = len(a)
        low, high = np.linalg.eigvalsh(a)[[0, -1]]
        c = sddm_factor(a)
        assert np.linalg.norm(c.T@c-a) < 1e-12
        d = math.ceil(128/epsilon**2*np.log(12*n*len(shifts)/failure))
        q = rng.normal(size=(d, n))/np.sqrt(d)
        p = rng.normal(size=(d, len(c)))@c/np.sqrt(d)
        for shift in shifts:
            aj = a+shift*np.eye(n)
            r = np.linalg.inv(aj)
            x, y = p@r, q@r
            # Deliberately inexact solves, scaled to the proved residual budgets.
            h_budget = epsilon*(low+shift)/(16*(high+shift))
            k_budget = epsilon*(low+shift)/(16*np.sqrt(high+shift))
            ey, ex = rng.normal(size=y.shape), rng.normal(size=x.shape)
            ey *= .8*h_budget/np.linalg.norm(ey@aj, "fro")
            ex *= .8*k_budget/np.linalg.norm(ex@aj, "fro")
            yh, xh = y+ey, x+ex
            hy, hx = np.linalg.norm(q-yh@aj, "fro"), np.linalg.norm(p-xh@aj, "fro")
            assert hy <= h_budget+1e-10 and hx <= k_budget+1e-10
            estimated2 = np.sum(yh*yh, axis=0)
            estimated1 = np.sum((xh+np.sqrt(shift)*yh)**2, axis=0)
            exact1, exact2 = np.diag(r), np.diag(r@r)
            error = max(np.max(abs(estimated1/exact1-1)), np.max(abs(estimated2/exact2-1)))
            max_error = max(max_error, error)
            assert error < epsilon
            # Saved relative intervals can seed deterministic T5 error propagation.
            assert np.all(estimated1/(1+epsilon) <= exact1)
            assert np.all(exact1 <= estimated1/(1-epsilon))
            assert np.all(estimated2/(1+epsilon) <= exact2)
            assert np.all(exact2 <= estimated2/(1-epsilon))
            lo1, hi1 = estimated1/(1+epsilon), estimated1/(1-epsilon)
            lo2, hi2 = estimated2/(1+epsilon), estimated2/(1-epsilon)
            for beta in [.2, 2, np.inf]:
                offset = 0 if beta == np.inf else 1/beta
                lower_margin, upper_margin = lo2/(offset+hi1), hi2/(offset+lo1)
                exact_margin = exact2/(offset+exact1)
                assert np.all(lower_margin <= exact_margin) and np.all(exact_margin <= upper_margin)
            # Energy-relative accuracy implies the stated residual bound.
            energy_error = np.sqrt(np.sum((ey@aj)*ey))
            assert hy <= np.sqrt(high+shift)*energy_error+1e-10
            batches += 1
    print(f"PASS T6 all-node initialization: {batches} factor/sketch/inexact-solve batches; "
          f"max relative error={max_error:.4f} < {epsilon}; residual tolerance=1e-10, factor tolerance=1e-12")


run("T6 all-node initialization", test)
