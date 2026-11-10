"""Permanent upper bounds, one fresh certification per candidate/round, and greedy guarantee."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase3_utils.py").read_text(), "_phase3_utils.py", "exec"))


def test():
    rng = np.random.default_rng(703)
    theta, cases, total_refreshes, max_depth = .12, 0, 0, 0
    instances = [(fixtures()[0], .8), (fixtures()[2], np.inf),
                 (2*np.eye(6), np.inf)]  # Includes exact ties and early breakdown.
    for a, penalty in instances:
        n = len(a)
        for power, heat in [(1, None), (2, None), (3, None), (.5, None), (None, .7)]:
            f, g = family_functions(power, heat)
            values = {z: trace_function(grounded(a, z, penalty), f) for z in subsets(n)}
            if power is not None and float(power).is_integer():
                keys, _ = coarse_jl_keys(a, penalty, int(power), rng)
            else:
                lower = .5*np.linalg.eigvalsh(a)[0]
                cap = f(lower) if penalty == np.inf else min(penalty*g(lower), f(lower))
                keys = np.full(n, cap)
            for budget in [1, min(3, n-1)]:
                chosen, refreshes, selections, pops = certified_celf(
                    a, penalty, budget, theta, keys, power, heat)
                assert len(set((j, u) for j, u, _, _ in refreshes)) == len(refreshes)
                assert len(refreshes) <= budget*n-budget*(budget-1)//2
                assert pops <= len(refreshes)+budget
                for base, u, lo, next_upper, snapshot in selections:
                    gains = {v: values[base]-values[base|{v}] for v in range(n) if v not in base}
                    assert all(gains[v] <= cap+2e-8*max(1, gains[v]) for v, cap in snapshot.items())
                    assert gains[u] >= (1-theta)*max(gains.values())-2e-8*max(1, max(gains.values()))
                    assert lo >= (1-theta)*next_upper-1e-12
                optimum = values[frozenset()]-min(val for z, val in values.items() if len(z) == budget)
                actual = values[frozenset()]-values[chosen]
                factor = 1-(1-(1-theta)/budget)**budget
                assert actual >= factor*optimum-2e-8*max(1, optimum)
                assert factor >= 1-1/np.e-theta
                total_refreshes += len(refreshes)
                max_depth = max(max_depth, max(r[2] for r in refreshes))
                cases += 1
    print(f"PASS T7 CELF: {cases} exhaustive-optimum comparisons, {total_refreshes} refreshes, max h={max_depth}; "
          "permanent bounds, per-round uniqueness, ties, and greedy factor pass; tolerance=2e-8")


run("T7 certified lazy greedy", test)
