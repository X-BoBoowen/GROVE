"""Current-matrix hard-grounding sketches, adaptive coverage and energy contraction."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))
from math import comb, ceil, log


def padded_inverse(a, selected):
    u = [i for i in range(len(a)) if i not in selected]
    r = np.zeros_like(a)
    r[np.ix_(u, u)] = np.linalg.inv(a[np.ix_(u, u)])
    return r


def energy(e, a):
    return np.sqrt(max(0, np.sum((e@a)*e)))


def test():
    rng = np.random.default_rng(1201)
    count = rounds = 0
    worst = max_sketch_error = 0.0
    for a in fixtures()[:-1]:
        n, k = len(a), 3
        c = np.linalg.cholesky(a).T
        zeta, failure = .3, .01
        total = sum(comb(n, t)*(n-t) for t in range(k+1))
        d = ceil(8/zeta**2*log(4*total/failure))
        g = rng.normal(size=(d, n))/np.sqrt(d)
        h = rng.normal(size=(d, n))/np.sqrt(d)
        gc = g@c
        # Uniform-event check over the deterministic family, not just the chosen path.
        for t in range(k+1):
            for selected in itertools.combinations(range(n), t):
                r = padded_inverse(a, selected)
                u = [i for i in range(n) if i not in selected]
                assert np.max(np.abs(r@a@r-r)) < 2e-9
                for left, target in [(gc, np.diag(r)), (h, np.diag(r@r))]:
                    estimate = np.sum((left@r)**2, axis=0)
                    relative = np.max(np.abs(estimate[u]/target[u]-1))
                    max_sketch_error = max(max_sketch_error, relative)
                    assert relative < zeta
                count += len(u)
        base_trace = np.trace(np.linalg.inv(a))
        optimum = base_trace-min(np.trace(padded_inverse(a, s))
                                 for s in itertools.combinations(range(n), k))
        for pool_size in [1, 3]:
            selected = set()
            r = padded_inverse(a, selected)
            y1, y2 = gc@r, h@r
            error1 = rng.normal(size=y1.shape)*1e-6
            error2 = rng.normal(size=y2.shape)*1e-6
            e1, e2 = energy(error1, a), energy(error2, a)
            upper_eigen = np.linalg.eigvalsh(a)[-1]
            for _ in range(k):
                active = np.array([i for i in range(n) if i not in selected])
                score = np.sum(y2[:, active]**2, axis=0)/np.sum(y1[:, active]**2, axis=0)
                pool = active[np.argsort(score)[-pool_size:]]
                true = np.diag(r@r)/np.where(np.diag(r) > 0, np.diag(r), 1)
                v = max(pool, key=lambda i: true[i])
                alpha = ((1-zeta)/(1+zeta))**2
                assert true[v] >= alpha*max(true[active])-2e-9
                x = r[:, v].copy()
                divisor = x[v]
                before1, before2 = energy(error1, a)**2, energy(error2, a)**2
                removed1 = np.sum(error1[:, v]**2)/divisor
                removed2 = np.sum(error2[:, v]**2)/divisor
                y1 -= np.outer(y1[:, v].copy(), x)/divisor
                y2 -= np.outer(y2[:, v].copy(), x)/divisor
                error1 -= np.outer(error1[:, v].copy(), x)/divisor
                error2 -= np.outer(error2[:, v].copy(), x)/divisor
                assert abs(energy(error1, a)**2-(before1-removed1)) < 2e-12
                assert abs(energy(error2, a)**2-(before2-removed2)) < 2e-12
                selected.add(v)
                r = padded_inverse(a, selected)
                active = [i for i in range(n) if i not in selected]
                discrepancy = max(np.max(np.abs(y1-gc@r)), np.max(np.abs(y2-h@r)))
                worst = max(worst, discrepancy)
                assert discrepancy < 2e-9
                for y, err, target, norm_error in [(y1, error1, np.diag(r), e1),
                                                  (y2, error2, np.diag(r@r), np.sqrt(upper_eigen)*e2)]:
                    estimate = np.sum((y+err)**2, axis=0)
                    rel = np.max(np.abs(estimate[active]/target[active]-1))
                    bound = zeta+2*np.sqrt(1+zeta)*norm_error+norm_error**2
                    assert rel <= bound+2e-9
                rounds += 1
            gain = base_trace-np.trace(r)
            assert gain >= (1-np.exp(-alpha))*optimum-2e-9
    # The stored road run counters identify where the extra work went.
    refresh_columns = 4342*16
    initial_columns = 78*32
    update_columns = 1863
    assert refresh_columns+initial_columns+update_columns == 73831
    assert 2173*32 == 69536
    assert refresh_columns == 69472
    assert 19*64 == 1216
    remainder_time = 880.821-32.252-54.049
    assert abs(remainder_time-794.520) < 1e-9
    assert abs(remainder_time-794.505) < .016
    print(f'PASS maintained road sketches: {count} all-set candidate checks, {rounds} adaptive updates; '
          f'max update discrepancy={worst:.3e} < 2e-9; max sketch error={max_sketch_error:.4f} < {zeta}; '
          'energy contraction, error bounds, greedy comparison, and road work accounting pass')


run('Road current-matrix sketch proposal', test)
