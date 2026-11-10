"""Corollary 3: positive Laplace representations and CFCC electrical identity."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))
from scipy.integrate import quad
from scipy.special import gamma


def test():
    worst = 0.0
    for s in [.1, .5, 1, 1.5, 3, 7]:
        for x in [.2, 1, 4]:
            # Substitute u=x*t and u=z**(1/s) to remove the origin singularity.
            value = quad(lambda z: np.exp(-z**(1/s))/s, 0, np.inf,
                         epsabs=1e-10, epsrel=1e-10, limit=400)[0] / gamma(s) / x**s
            err = abs(value*x**s - 1)
            worst = max(worst, err)
            assert err < 2e-9
    l = laplacian(5, [(0, 1, .5), (1, 2, 2), (2, 3, .8), (3, 4, 1.3), (0, 4, .4)])
    for sigma in [0, .3]:
        seed = {0} if sigma == 0 else set()
        a = l + sigma*np.eye(5)
        for mode in [.7, np.inf]:
            for f in [lambda x: x**(-4), lambda x: np.exp(-.8*x), lambda x: 1/(x+.2)]:
                values = {s: trace_function(grounded(a, s, mode), f)
                          for s in subsets(5) if seed <= s}
                check_set_function(values, 5, 1e-9)
    # Unit-current voltage at each source, Dirichlet zero voltages on S.
    for s in subsets(5):
        if not s or len(s) == 5:
            continue
        b = grounded(l, s, np.inf)
        resistance_sum = sum(np.linalg.solve(b, np.eye(len(b))[:, j])[j]
                             for j in range(len(b)))
        inverse_trace = trace_function(b, lambda x: 1/x)
        assert abs(resistance_sum - inverse_trace) < 1e-10
        assert abs(5/resistance_sum - 5/inverse_trace) < 1e-10
    print(f"PASS T1 corollaries: Laplace relative error={worst:.3e} < 2e-9; "
          "set inequalities tolerance=1e-9; CFCC voltage/trace tolerance=1e-10")


run("T1 corollaries", test)
