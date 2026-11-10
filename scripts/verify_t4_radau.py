"""Single-run Gauss/Radau CM bounds, penalty integration and hard-tail certificate."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase2_utils.py").read_text(), "_phase2_utils.py", "exec"))
from scipy.integrate import quad


def test():
    cases, worst = 0, 0.0
    pairs = [(lambda x: x**(-.4), lambda x: .4*x**(-1.4), 0),
             (lambda x: x**(-2), lambda x: 2*x**(-3), 0),
             (lambda x: np.exp(-.8*x), lambda x: .8*np.exp(-.8*x), 0),
             (lambda x: 2+1/x+np.exp(-x), lambda x: x**(-2)+np.exp(-x), 2)]
    for a in fixtures()[:3]:
        lower = .8*np.linalg.eigvalsh(a)[0]
        for depth in [1, 2, len(a)]:
            _, t, gamma = lanczos(a, 0, depth)
            for penalty in [0, .1, 2, 100]:
                _, independent_t, _ = lanczos(updated(a, 0, penalty), 0, depth)
                tu = updated(t, 0, penalty)
                assert np.linalg.norm(independent_t-tu) < 2e-8
                ru = radau(t, gamma, lower, penalty)
                if gamma:
                    assert abs(np.linalg.eigvalsh(ru)[0]-lower) < 1e-9
                for f, g, _ in pairs:
                    exact = diagonal_function(updated(a, 0, penalty), g)
                    lo, hi = diagonal_function(tu, g), diagonal_function(ru, g)
                    tol = 2e-9*max(1, exact, hi)
                    assert lo <= exact+tol and exact <= hi+tol
                    worst = max(worst, max(lo-exact, exact-hi, 0)/max(1, exact))
                    cases += 1
            for f, g, atom in pairs:
                for penalty in [2, np.inf]:
                    exact = trace_function(a, f)-trace_function(updated(a, 0, penalty), f)
                    reduced = trace_function(t, f)-trace_function(updated(t, 0, penalty), f)
                    assert reduced <= exact+2e-8*max(1, exact)
                    cap = 20.0 if penalty == np.inf else penalty
                    # A certified left-endpoint upper sum uses monotonicity of
                    # the TRUE penalty integrand; Radau itself need not be monotone.
                    grid = np.linspace(0, cap, 81)
                    upper = sum((right-left)*diagonal_function(radau(t, gamma, lower, left), g)
                                for left, right in zip(grid[:-1], grid[1:]))
                    if penalty == np.inf:
                        d = a[0, 0]
                        variance = np.sum(a[0, 1:]**2)
                        tail = 4*variance*g(lower)/(d+cap)+2*(f((d+cap)/2)-atom)
                        upper += atom+tail
                        actual_tail = trace_function(updated(a, 0, cap), f)
                        actual_tail -= trace_function(updated(a, 0, np.inf), f)+atom
                        assert actual_tail <= tail+1e-8
                    assert exact <= upper+2e-8*max(1, exact)
                    # Numerical integration checks the exact reduced penalty identity.
                    integral = quad(lambda u: diagonal_function(updated(t, 0, u), g),
                                    0, penalty, epsabs=1e-8, epsrel=1e-9, limit=250)[0]
                    if penalty == np.inf:
                        integral += atom
                    assert abs(integral-reduced) < 2e-7*max(1, reduced)
    print(f"PASS T4 Gauss/Radau: {cases} CM sandwiches; max scaled violation={worst:.3e}; "
          "point tolerance=2e-9, integrated/tail tolerance=2e-8, identity tolerance=2e-7")


run("T4 Gauss/Radau", test)
