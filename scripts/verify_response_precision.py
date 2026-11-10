"""Checks cache absolute-error drift and candidate-specific deflation bounds."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def test():
    count = 0
    for t in [.1, .01, .001, .0001]:
        x = np.array([[1+t, -1.], [-1., 1.]])
        true0 = np.linalg.inv(x)[1, 1]
        true1 = 1.0
        estimated0 = .99*true0
        estimated1 = estimated0-(true0-true1)
        assert abs((estimated1-true1)-(estimated0-true0)) < 1e-10
        assert abs(estimated1-true1) > .01/t
        if t <= .001:
            assert estimated1 < 0
    # Full update recurrence, allowing arbitrary column/round errors.
    rng = np.random.default_rng(812)
    true = 100.0
    estimated = 101.0
    expected_error = 1.0
    for _ in range(20):
        decrement = rng.uniform(0, 2)
        perturbation = rng.uniform(-.02, .02)
        true -= decrement
        estimated -= decrement+perturbation
        expected_error -= perturbation
        assert abs(estimated-true-expected_error) < 1e-11
    max_effective = 0.0
    for x in fixtures():
        vals, vecs = np.linalg.eigh(x)
        for r in range(1, len(x)):
            for j in [1, 2, 4, 8]:
                weights = vecs**2
                head = weights[:, :r] @ vals[:r]**(-j)
                tail = weights[:, r:] @ vals[r:]**(-j)
                full = head+tail
                mass = weights[:, :r].sum(axis=1)
                bound = np.minimum(1, (1-mass)/mass*(vals[r-1]/vals[r])**j)
                fraction = tail/full
                assert np.all(fraction <= bound+2e-10)
                perturbed_tail = tail*(1+rng.uniform(-.2, .2, len(x)))
                rel = np.abs(head+perturbed_tail-full)/full
                assert np.all(rel <= .2*fraction+2e-10)
                max_effective = max(max_effective, rel.max())
                count += len(x)
    # Even with connected followers, a small spectral ratio alone is insufficient.
    for j in range(1, 9):
        t = .1
        b = t**(j+2)
        x = np.array([[t+b, -b], [-b, 1+b]])
        vals, vecs = np.linalg.eigh(x)
        head = vecs[1, 0]**2 * vals[0]**(-j)
        tail = vecs[1, 1]**2 * vals[1]**(-j)
        assert tail/(head+tail) > .9999
        assert (vals[0]/vals[1])**j < .102**j
    print(f'PASS precision conditions: {count} vertex/rank/moment checks; cache recurrence and '
          f'negative-cache controls; 8 connected deflation controls; tolerance=2e-10; '
          f'max effective sketch error={max_effective:.4f}')


run('Cache and deflation precision', test)
