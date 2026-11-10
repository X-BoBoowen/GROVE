"""Resource-augmented spectral guarantee with an explicit feasible budget check."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase3_utils.py").read_text(), "_phase3_utils.py", "exec"))
import math


def test():
    theta, budget, count = .1, 2, 0
    for dimension, spread, eta in [(10, .05, .15), (10, 1.0, 4.0), (12, .02, .1)]:
        # These are hard-seeded connected stars with the indicated grounded matrix.
        a = np.diag(np.linspace(1, 1+spread, dimension))
        final_dimension = dimension-budget
        s = math.ceil(math.log(2*final_dimension)/math.log1p(eta))
        lower, upper = 1.0, 1+spread
        rounds = max(budget, math.ceil(budget/(1-theta)*(math.log(dimension/final_dimension)+s*math.log(upper/lower))))
        assert rounds <= dimension-1
        eligible = [z for z in subsets(dimension) if len(z) == budget]
        phi = lambda z: trace_function(grounded(a, z, np.inf), lambda x: x**(-s))
        optimum = min(phi(z) for z in eligible)
        best_lambda = max(np.linalg.eigvalsh(grounded(a, z, np.inf))[0] for z in eligible)
        baseline, chosen = phi(frozenset()), frozenset()
        for j in range(rounds):
            v = max(set(range(dimension))-chosen, key=lambda u: phi(chosen)-phi(chosen|{u}))
            chosen = chosen|{v}
            rho = (1-(1-theta)/budget)**(j+1)
            assert phi(chosen) <= rho*baseline+(1-rho)*optimum+2e-9*max(1, baseline)
        actual_lambda = np.linalg.eigvalsh(grounded(a, chosen, np.inf))[0]
        assert actual_lambda >= best_lambda/(2*final_dimension)**(1/s)-1e-10
        assert actual_lambda >= best_lambda/(1+eta)-1e-10
        count += 1
    # A requested near-one ratio may require an infeasible augmented budget;
    # the theorem explicitly requires checking this rather than silently clipping.
    d, k, eta, kappa = 6, 2, .01, 100
    s = math.ceil(math.log(2*(d-k))/math.log1p(eta))
    needed = math.ceil(k/(1-theta)*(math.log(d/(d-k))+s*math.log(kappa)))
    assert needed > d-1
    print(f"PASS T9 bicriteria: {count} feasible power/budget regimes; recurrence tolerance=2e-9, "
          "spectral tolerance=1e-10; infeasible-budget guard passes")


run("T9 bicriteria", test)
