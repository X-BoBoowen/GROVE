"""Conditional practical inc decisions with persistent initial error and exact updates."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_phase3_utils.py').read_text(), '_phase3_utils.py', 'exec'))
import heapq


def all_moments(base, selected, order):
    alive = [i for i in range(len(base)) if i not in selected]
    current = base[np.ix_(alive, alive)]
    eig, vec = np.linalg.eigh(current)
    result = np.zeros((order, len(base)))
    for j in range(1, order+1):
        result[j-1, alive] = vec**2 @ eig**(-j)
    return result


def test():
    rng = np.random.default_rng(1105)
    cases = steps = 0
    smallest_gain = 1.0
    for base in fixtures()[:-1]:
        n = len(base)
        base = base+np.eye(n)*.2
        k = min(3, n-2)
        for p in [2, 3, 4]:
            jmax = p+1
            initial = all_moments(base, set(), jmax)
            minimum = initial.copy()
            for size in range(k+1):
                for ss in itertools.combinations(range(n), size):
                    m = all_moments(base, set(ss), jmax)
                    alive = [i for i in range(n) if i not in ss]
                    minimum[:, alive] = np.minimum(minimum[:, alive], m[:, alive])
            gamma_retention = np.min(minimum/initial)
            epsilon, theta = .2, .1
            zeta0 = gamma_retention*epsilon/(128*p)
            opt_residual = min(trace_function(updated_set(base, set(s)), lambda x: x**(-p))
                               for s in itertools.combinations(range(n), k))
            base_value = trace_function(base, lambda x: x**(-p))
            opt_gain = base_value-opt_residual
            for slack in [0.0, .2]:
                selected, refreshed = set(), set()
                cache = initial*(1+rng.uniform(-zeta0, zeta0, initial.shape))
                for round_id in range(k):
                    true = all_moments(base, selected, jmax)
                    alive = [i for i in range(n) if i not in selected]
                    relative = np.abs(cache[:, alive]/true[:, alive]-1)
                    assert relative.max() <= epsilon/(64*p)+3e-9
                    def marginal(v):
                        return moment_marginal(cache[:, v], np.inf, p)
                    heap = [(-marginal(v)*(1 if v in refreshed else 1+slack), v) for v in alive]
                    heapq.heapify(heap)
                    best, value = None, -np.inf
                    while True:
                        while heap and heap[0][1] in refreshed:
                            key, v = heapq.heappop(heap)
                            if -key > value:
                                best, value = v, -key
                        upper_key = -heap[0][0] if heap else 0
                        if best is not None and value >= (1-theta)*upper_key:
                            break
                        _, v = heapq.heappop(heap)
                        assert v not in refreshed
                        refreshed.add(v)
                        cache[:, v] = true[:, v]
                        val = marginal(v)
                        if val > value:
                            best, value = v, val
                    true_gains = {v: moment_marginal(true[:, v], np.inf, p) for v in alive}
                    assert true_gains[best] >= (1-epsilon)*max(true_gains.values())-3e-9*base_value
                    selected.add(best)
                    after = all_moments(base, selected, jmax)
                    cache -= true-after
                    # Bounded solve/update perturbations; the initial JL error is not redrawn.
                    noise = rng.uniform(-1, 1, cache.shape)*minimum*epsilon/(128*p*k)
                    cache += noise
                    steps += 1
                residual = trace_function(updated_set(base, selected), lambda x: x**(-p))
                gain = base_value-residual
                smallest_gain = min(smallest_gain, gain/opt_gain)
                assert gain >= (1-np.exp(-1)-epsilon)*opt_gain-3e-9*base_value
                cases += 1
    print(f'PASS T11 practical guarantee: {cases} adaptive runs, {steps} rounds; persistent bounded '
          f'initial errors, update noise and exhaustive OPT checks; min gain/OPT={smallest_gain:.6f}; tolerance=3e-9')


def updated_set(base, selected):
    alive = [i for i in range(len(base)) if i not in selected]
    return base[np.ix_(alive, alive)]


run('T11 conditional practical inc', test)
