"""Spectral-shift moment matching, dimension-free width, and heat parameter dependence."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase2_utils.py").read_text(), "_phase2_utils.py", "exec"))


def test():
    count = 0
    for a in fixtures()[:3]:
        low, high = .8*np.linalg.eigvalsh(a)[0], 1.01*np.linalg.eigvalsh(a)[-1]
        for depth in [1, 2, 3]:
            _, t, _ = lanczos(a, 0, depth)
            h = len(t)
            for penalty in [.7, np.inf]:
                b, tb = updated(a, 0, penalty), updated(t, 0, penalty)
                max_degree = 2*h if penalty != np.inf else 2*h-1
                for degree in range(max_degree+1):
                    f = lambda x: x**degree
                    exact = trace_function(a, f)-trace_function(b, f)
                    small = trace_function(t, f)-trace_function(tb, f)
                    scale = max(1, trace_function(a, lambda x: abs(x)**degree), abs(small))
                    assert abs(exact-small) <= 2e-9*scale
                end = high+(0 if penalty == np.inf else penalty)
                rho = (np.sqrt(2*end-low)+np.sqrt(low))/(np.sqrt(2*end-low)-np.sqrt(low))
                degree = 2*h-2 if penalty == np.inf else 2*h-1
                for f, g in [(lambda x: x**(-.4), lambda x: .4*x**(-1.4)),
                             (lambda x: x**(-2), lambda x: 2*x**(-3)),
                             (lambda x: np.exp(-x), lambda x: np.exp(-x))]:
                    exact = trace_function(a, f)-trace_function(b, f)
                    small = trace_function(t, f)-trace_function(tb, f)
                    poly_error = 2*g(low/2)*rho**(-degree)/(rho-1)
                    width = (high-low)*poly_error if penalty == np.inf else 2*penalty*poly_error
                    assert -2e-8*max(1, exact) <= exact-small <= width+2e-8*max(1, exact)
                    count += 1
    # Bounded condition number does not bound relative heat convergence uniformly
    # in tau: all finite h<n miss the slower smallest-eigenvalue decay rate.
    ratios = []
    for n in [8, 16, 32]:
        a = 2*np.eye(n)-.5*np.diag(np.ones(n-1), 1)-.5*np.diag(np.ones(n-1), -1)
        _, t, _ = lanczos(a, 0, 3)
        assert np.linalg.cond(a) < 3
        heat = lambda x: np.exp(-200*x)
        exact = trace_function(a, heat)-trace_function(a[1:, 1:], heat)
        small = trace_function(t, heat)-trace_function(t[1:, 1:], heat)
        ratios.append(small/exact)
        assert small/exact < 1e-12
    print(f"PASS T4 convergence: {count} width checks and polynomial moments; tolerance=2e-8; "
          f"fixed-kappa heat controls max lower/exact={max(ratios):.3e} < 1e-12")


run("T4 convergence", test)
