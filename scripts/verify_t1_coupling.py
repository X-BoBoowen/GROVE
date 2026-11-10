"""Lemma 1: independent local-time clocks, killed kernels, all-set coupling."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))
from scipy.linalg import expm
from scipy.stats import poisson


def test():
    rng = np.random.default_rng(2718)
    l = laplacian(4, [(0, 1, 0.7), (1, 2, 1.1), (2, 3, 0.4), (0, 3, 0.3)])
    sigma, beta, t = 0.2, 1.7, 0.8
    a = l + sigma * np.eye(4)
    sets = subsets(4)
    max_error = 0.0
    # Uniformization is the sum over independent Poisson event counts and
    # substochastic jump/kill transitions, including starts already grounded.
    for mode in [beta, np.inf]:
        for s in sets:
            b = grounded(a, s, mode)
            if not len(b):
                continue
            rate = max(np.diag(b))
            p = np.eye(len(b)) - b / rate
            assert p.min() >= -1e-15 and np.max(p.sum(axis=1)) <= 1 + 1e-15
            for time in [0.0, 0.03, 0.8, 4.0]:
                jmax = int(poisson.isf(1e-14, rate * time)) if time else 0
                power, kernel = np.eye(len(b)), np.zeros_like(b)
                for j in range(jmax + 1):
                    kernel += poisson.pmf(j, rate * time) * power
                    power = power @ p
                error = np.max(np.abs(kernel - expm(-time * b)))
                max_error = max(max_error, error)
                assert error < 3e-13
    # Independent simulation of the base walk and Exp clocks in LOCAL time.
    count = 40000
    empirical = {mode: np.zeros((len(sets), 4)) for mode in [beta, np.inf]}
    max_mc_error = 0.0
    for start in range(4):
        for _ in range(count):
            local, visited = np.zeros(4), {start}
            now, state = 0.0, start
            while now < t:
                wait = rng.exponential(1 / l[state, state])
                local[state] += min(wait, t - now)
                now += wait
                if now < t:
                    probs = -l[state].copy()
                    probs[state] = 0
                    state = rng.choice(4, p=probs / probs.sum())
                    visited.add(state)
            clocks = rng.exponential(1 / beta, 4)
            alive = rng.exponential(1 / sigma) > t
            for mode in [beta, np.inf]:
                bad = set(np.flatnonzero(clocks <= local)) if mode != np.inf else visited
                # This entire vector uses the SAME walk, base clock and node clocks.
                survival = {s: int(alive and not (s & bad)) for s in sets}
                check_set_function(survival, 4, 0)
                for j, s in enumerate(sets):
                    empirical[mode][j, start] += survival[s] * (state == start)
    for mode in [beta, np.inf]:
        for j, s in enumerate(sets):
            b = grounded(a, s, mode)
            target = np.zeros(4)
            keep = [i for i in range(4) if mode != np.inf or i not in s]
            target[keep] = np.diag(expm(-t * b)) if len(b) else []
            err = np.abs(empirical[mode][j] / count - target)
            tol = 8 * np.sqrt(target * (1 - target) / count) + 8 / count
            assert np.all(err <= tol), (mode, s, err, tol)
            max_mc_error = max(max_mc_error, np.max(err))
    print(f"PASS T1 coupling: uniformization error={max_error:.3e} < 3e-13; "
          f"clock MC max error={max_mc_error:.3e}, tolerance=8 SE+8/N; "
          f"pathwise violations=0 (exact tolerance=0), N={count}/start")


run("T1 coupling", test)
