"""Correct baseline-dependent spectral translation and an exact connected-star control."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase3_utils.py").read_text(), "_phase3_utils.py", "exec"))


def test():
    theta, cases = .12, 0
    for a, penalty in [(fixtures()[0], .8), (fixtures()[2], np.inf)]:
        n, budget = len(a), 2
        dimension = n if penalty != np.inf else n-budget
        eligible = [z for z in subsets(n) if len(z) == budget]
        best_lambda = max(np.linalg.eigvalsh(grounded(a, z, penalty))[0] for z in eligible)
        gamma = 1-(1-(1-theta)/budget)**budget
        rho = 1-gamma
        for s in [1, 2, 4]:
            f = lambda x: x**(-s)
            baseline = trace_function(a, f)
            optimum_phi = min(trace_function(grounded(a, z, penalty), f) for z in eligible)
            chosen, _, _, _ = certified_celf(a, penalty, budget, theta,
                                            np.full(n, (.5*np.linalg.eigvalsh(a)[0])**(-s)), power=s)
            actual_phi = trace_function(grounded(a, chosen, penalty), f)
            actual_lambda = np.linalg.eigvalsh(grounded(a, chosen, penalty))[0]
            assert actual_phi <= rho*baseline+gamma*optimum_phi+1e-8
            first_bound = best_lambda/(rho*baseline*best_lambda**s+gamma*dimension)**(1/s)
            contrast = baseline/optimum_phi
            contrast_bound = best_lambda/(dimension*(rho*contrast+gamma))**(1/s)
            assert actual_lambda >= first_bound-1e-9 and first_bound >= contrast_bound-1e-9
            cases += 1
    # Connected weighted star; root 0 is the fixed seed. Both outputs use two
    # additional anchors. This challenges the implication from gain alone,
    # not the actual CELF sequence or the CM theorem.
    smallest_ratio = 1.0
    for b in [.2, .01, 1e-5]:
        l = laplacian(4, [(0, 1, b/2), (0, 2, b), (0, 3, 1)])
        for s in [1, 2, 4, 12]:
            baseline = trace_function(grounded(l, {0}, np.inf), lambda x: x**(-s))
            good = trace_function(grounded(l, {0, 1, 2}, np.inf), lambda x: x**(-s))
            bad = trace_function(grounded(l, {0, 1, 3}, np.inf), lambda x: x**(-s))
            ratio = (baseline-bad)/(baseline-good)
            assert ratio >= 2/3-1e-12 and ratio > 1-1/np.e
            assert abs(ratio-(2**s+b**s)/(2**s+1)) < 2e-12
            spectral_ratio = np.linalg.eigvalsh(grounded(l, {0, 1, 3}, np.inf))[0]
            assert spectral_ratio == b
            assert spectral_ratio < (1-1/np.e)**(1/s)
            smallest_ratio = min(smallest_ratio, spectral_ratio)
    for gamma in [.9, .99]:
        m = int(np.ceil(max(2, gamma/(1-gamma))))
        for s in [1, 2, 5]:
            b = .001
            gain_ratio = (m**s+b**s)/(m**s+1)
            assert gain_ratio >= gamma-1e-12
    print(f"PASS T9 gain translation: {cases} CELF spectral bounds; tolerance=1e-9; "
          f"expected star control satisfies gain >=2/3 but lambda ratio={smallest_ratio:.1e}; exact formula tolerance=2e-12")


run("T9 gain translation", test)
