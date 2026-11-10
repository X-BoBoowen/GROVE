"""Fixed positive composite quadrature; scalar derivative and marginal relative guarantees."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase2_utils.py").read_text(), "_phase2_utils.py", "exec"))
import math


def test():
    eta, worst, count = .02, 0.0, 0
    for s in [.1, .5, .99, 1, 1.5, 2, 5]:
        m = int(np.floor(s))+1
        order = m+3+math.ceil(math.log(2/eta, 4))
        for kappa in [1, 10, 1e4]:
            low, high = .3, .3*kappa
            m, nodes, weights = power_quadrature(s, low, high, order)
            assert np.all(nodes > 0) and np.all(weights > 0)
            x = np.geomspace(low, high, 151)
            f, g = quadrature_values(x, m, nodes, weights)
            err = max(np.max(abs(f*x**s-1)), np.max(abs(g/(s*x**(-s-1))-1)))
            worst = max(worst, err)
            assert err <= eta/2+1e-9
            assert len(nodes) <= (math.ceil(math.log2(kappa))+2)*order
            count += 1
    l = laplacian(4, [(0, 1, .6), (1, 2, 1.1), (2, 3, .7), (0, 3, .5)])
    a = l+.4*np.eye(4)
    low, high = .4, np.linalg.eigvalsh(a)[-1]+2
    for s in [.2, .8, 1.5, 3]:
        m = int(np.floor(s))+1
        certified_order = m+3+math.ceil(math.log(2/eta, 4))
        for order in [1, 2, certified_order]:
            m, nodes, weights = power_quadrature(s, low, high, order)
            surrogate = lambda x: quadrature_values(x, m, nodes, weights)[0]/(1+eta/2)
            for penalty in [2, np.inf]:
                exact = {z: trace_function(grounded(a, z, penalty), lambda x: x**(-s)) for z in subsets(4)}
                values = {z: trace_function(grounded(a, z, penalty), surrogate) for z in subsets(4)}
                check_set_function(values, 4, 3e-9)
                if order == certified_order:
                    for z in subsets(4):
                        for v in set(range(4))-z:
                            drop, approx = exact[z]-exact[z|{v}], values[z]-values[z|{v}]
                            assert (1-eta)*drop-3e-8*max(1, drop) <= approx <= drop+3e-8*max(1, drop)
    # Explicit seed-dependent spectral bounds used to fix one rule globally.
    n, wmin = 4, .5
    for beta in [.1, 2, np.inf]:
        lower_bound = wmin/n**2 if beta == np.inf else 1/(2*n**2/wmin+2*n/beta)
        upper_bound = 2*max(np.diag(l))+(0 if beta == np.inf else beta)
        for selected in subsets(n):
            if 0 not in selected:
                continue
            b = grounded(l, selected, beta)
            if len(b):
                spectrum = np.linalg.eigvalsh(b)
                assert lower_bound <= spectrum[0]+1e-12 and spectrum[-1] <= upper_bound+1e-12
    print(f"PASS T4 fixed positive quadrature: {count} scalar grids; max relative f/g error={worst:.3e} "
          f"<= eta/2={eta/2}; all-order supermodularity tolerance=3e-9; marginal tolerance=3e-8")


run("T4 positive quadrature", test)
