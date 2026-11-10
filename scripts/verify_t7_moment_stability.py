"""Certified error transfer from independent relative moment errors to q_m."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase3_utils.py").read_text(), "_phase3_utils.py", "exec"))


def test():
    rng = np.random.default_rng(702)
    cases, max_fraction = 0, 0.0
    eta = .15
    for a in fixtures():
        v = len(a)//2
        for order in range(1, 9):
            rs = moments(a, v, order+1)
            epsilon = required_moment_precision(order, eta)
            proved = moment_error_bound(order, epsilon)
            assert abs(proved-eta) < 1e-12
            signs = (np.array(list(itertools.product([-1, 1], repeat=order+1))) if order <= 5
                     else rng.uniform(-1, 1, (100, order+1)))
            for penalty in [.1, 3, np.inf]:
                exact = moment_marginal(rs, penalty, order)
                assert exact > 0
                for sign in signs:
                    perturbed = rs*(1+epsilon*sign)
                    approximate = moment_marginal(perturbed, penalty, order)
                    error = abs(approximate/exact-1)
                    max_fraction = max(max_fraction, error/proved)
                    assert error <= proved+2e-9
                    assert approximate/(1+eta) <= exact+1e-8*max(1, exact)
                    assert exact <= approximate/(1-eta)+1e-8*max(1, exact)
                    # A coarser upper bound needs only endpoint moments, not the subtraction.
                    lower_r1, upper_rmp1 = perturbed[0]/(1+epsilon), perturbed[-1]/(1-epsilon)
                    offset = 0 if penalty == np.inf else 1/penalty
                    coarse = order*upper_rmp1/(offset+lower_r1)
                    assert exact <= coarse+1e-8*max(1, exact)
                    cases += 1
    print(f"PASS T7 moment stability: {cases} independent perturbations; max actual/proved relative error={max_fraction:.4f} <= 1; "
          "tolerance=2e-9; tight and cancellation-free coarse heap bounds pass")


run("T7 moment stability", test)
