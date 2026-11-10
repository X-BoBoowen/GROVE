"""Fresh-round Gaussian all-node greedy with exhaustive small-instance optima.

At the theorem's conservative sketch dimension, generate the exact Gaussian
Gram distribution by Bartlett's Wishart construction instead of materializing
millions of rows. This preserves the joint distribution of all squared-column
estimators and all moment orders. Dense inverse products are test references,
not an implementation/benchmark of the nearly-linear SDDM primitive.
"""
from pathlib import Path
exec(compile(Path(__file__).with_name('_phase3_utils.py').read_text(), '_phase3_utils.py', 'exec'))


def gaussian_gram(n, d, rng):
    t = np.zeros((n, n))
    for j in range(n):
        t[j, j] = np.sqrt(rng.chisquare(d-j))
        t[j, :j] = rng.normal(size=j)
    return (t@t.T)/d


def allnode_moments(a, order, d, rng):
    c = np.linalg.cholesky(a).T
    gq = gaussian_gram(len(a), d, rng)
    gp = c.T@gaussian_gram(len(a), d, rng)@c
    r, rpower = np.linalg.inv(a), np.eye(len(a))
    result = np.empty((len(a), order))
    for ell in range(1, (order+1)//2+1):
        rpower = rpower@r
        if 2*ell-1 <= order:
            result[:, 2*ell-2] = np.diag(rpower@gp@rpower)
        if 2*ell <= order:
            result[:, 2*ell-1] = np.diag(rpower@gq@rpower)
    return result, gq, gp


def test():
    epsilon, failure = .4, .05
    root = np.random.SeedSequence(7102)
    cases, rounds, max_moment, min_gain = 0, 0, 0., 1.
    for initial in fixtures():
        for penalty in [.7, np.inf]:
            for p in [1, 2, 4, 8]:
                n = len(initial)
                budget = min(3, n)
                zeta, eta = epsilon/(32*p), epsilon/2
                d = int(np.ceil(128/zeta**2*np.log(12*n*(p+1)*budget/failure)))
                f = lambda x: x**(-p)
                baseline = trace_function(initial, f)
                feasible = list(itertools.combinations(range(n), budget))
                best = max(baseline-trace_function(grounded(initial, s, penalty), f) for s in feasible)
                selected, seen_seeds = set(), set()
                for step in range(budget):
                    a = grounded(initial, selected, penalty)
                    eligible = [u for u in range(n) if u not in selected]
                    child = root.spawn(1)[0]
                    assert child.spawn_key not in seen_seeds
                    seen_seeds.add(child.spawn_key)
                    est, _, _ = allnode_moments(a, p+1, d, np.random.default_rng(child))
                    estimates, exact = [], []
                    for v in eligible:
                        u = eligible.index(v) if penalty == np.inf else v
                        rs = moments(a, u, p+1)
                        err = max(abs(est[u]/rs-1))
                        max_moment = max(max_moment, err)
                        assert err <= zeta+2e-10
                        value = positive_integer_marginal(a, u, penalty, p)[0]
                        approx = moment_marginal(est[u], penalty, p)
                        assert abs(approx/value-1) <= eta+2e-8
                        estimates.append(approx)
                        exact.append(value)
                    j = int(np.argmax(estimates))
                    assert exact[j] >= (1-eta)/(1+eta)*max(exact)-2e-8*max(exact)
                    selected.add(eligible[j])
                    rounds += 1
                gain = baseline-trace_function(grounded(initial, selected, penalty), f)
                factor = 1-(1-(1-epsilon)/budget)**budget
                assert gain >= factor*best-2e-8*max(1., best)
                min_gain = min(min_gain, gain/best)
                cases += 1
    # Probability argument is a conditional union bound, not a product of
    # independent success events. Check worst conditional budgets numerically.
    for k in [1, 2, 20, 1000]:
        epsilons = np.full(k, failure/k)
        assert np.sum(epsilons) <= failure+1e-14
        assert 1-np.prod(1-epsilons) <= failure+1e-14
    print(f'PASS T7b fresh greedy: {cases} exhaustive-optimum comparisons, {rounds} fresh rounds; '
          f'max relative moment error={max_moment:.5g}; min gain/OPT={min_gain:.5f}; tolerance=2e-8')


run('T7b fresh greedy', test)
