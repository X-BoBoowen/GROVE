"""Deterministic error envelopes survive adaptive selection and sequential solves."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))


def test():
    rng = np.random.default_rng(5801)
    cases, max_ratio = 0, 0.0
    for beta in [.3, 3, np.inf]:
        for shift in [0, .7]:
            a = fixtures()[0]+shift*np.eye(6)
            r = np.linalg.inv(a)
            first, second = np.diag(r).copy(), np.diag(r@r).copy()
            e1, e2 = np.full(6, 1e-5), np.full(6, 2e-5)
            first += rng.uniform(-1, 1, 6)*e1
            second += rng.uniform(-1, 1, 6)*e2
            available = list(range(6))
            for step in range(4):
                low, high = np.linalg.eigvalsh(a)[[0, -1]]
                first = np.clip(first, 1/high, 1/low)
                second = np.clip(second, 1/high**2, 1/low**2)
                candidates = range(len(a)) if beta == np.inf else available
                v = max(candidates, key=lambda j: second[j]/first[j])
                x, y = r[:, v], (r@r)[:, v]
                xhat = np.clip(x+rng.normal(0, 1e-7, len(a)), 0, 1/low)
                yhat = np.clip(y+rng.normal(0, 1e-7, len(a)), 0, 1/low**2)
                rho_x = np.linalg.norm(np.eye(len(a))[:, v]-a@xhat)
                rho_y = np.linalg.norm(xhat-a@yhat)
                ex, ey = rho_x/low, (rho_y+rho_x/low)/low
                assert np.linalg.norm(xhat-x) <= ex+1e-12
                assert np.linalg.norm(yhat-y) <= ey+1e-12
                delta = (0 if beta == np.inf else 1/beta)+1/high
                c = 1/((0 if beta == np.inf else 1/beta)+first[v])
                ec, cmax, xmax, ymax, zmax = e1[v]/delta**2, 1/delta, 1/low, 1/low**2, 1/low**2
                d1 = ec*xmax**2+2*cmax*xmax*ex
                d2 = (2*ec*xmax*ymax+2*cmax*(ex*ymax+xmax*ey)
                      +2*cmax*ec*zmax*xmax**2+cmax**2*e2[v]*xmax**2
                      +2*cmax**2*zmax*xmax*ex)
                new_first = first-c*xhat**2
                new_second = second-2*c*xhat*yhat+c*c*second[v]*xhat**2
                e1, e2 = e1+d1, e2+d2
                a = updated(a, v, beta)
                r = np.linalg.inv(a)
                if beta == np.inf:
                    keep = np.arange(len(first)) != v
                    new_first, new_second, e1, e2 = [z[keep] for z in (new_first, new_second, e1, e2)]
                else:
                    available.remove(v)
                first, second = new_first, new_second
                error1, error2 = abs(first-np.diag(r)), abs(second-np.diag(r@r))
                assert np.all(error1 <= e1+1e-10) and np.all(error2 <= e2+1e-10)
                max_ratio = max(max_ratio, np.max(error1/e1), np.max(error2/e2))
                cases += 1
    print(f"PASS T5 cache errors: {cases} adaptive rounds; max actual/envelope={max_ratio:.4f} <= 1; "
          "arithmetic tolerance=1e-10, solve-bound tolerance=1e-12")


run("T5 cache error propagation", test)
