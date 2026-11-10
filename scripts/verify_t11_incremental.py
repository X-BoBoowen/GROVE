"""All-order independent decrements, column envelopes, and unique-refresh counts."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def mul(a, b):
    vals = np.array([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]])
    return np.array([vals.min(axis=0), vals.max(axis=0)])


def sub(a, b):
    return np.array([a[0]-b[1], a[1]-b[0]])


def decrement_intervals(a, c, s, jmax):
    gamma = [a[0]]
    for t in range(1, jmax):
        total = sum((mul(s[i], gamma[t-1-i]) for i in range(t)), np.zeros_like(a[0]))
        gamma.append(sub(a[t], mul(c, total)))
    drops = []
    for j in range(1, jmax+1):
        total = sum((mul(a[i], gamma[j-1-i]) for i in range(j)), np.zeros_like(a[0]))
        drops.append(mul(c, total))
    return drops


def test():
    rng = np.random.default_rng(1103)
    cases = 0
    worst = 0.0
    for matrix in fixtures()[:-1]:
        # Moderate scaling makes checks through power 9 well conditioned.
        xmat = matrix/np.linalg.norm(matrix, 2)+np.eye(len(matrix))
        r = np.linalg.inv(xmat)
        for v in range(len(r)):
            x = r[:, v]
            c = 1/r[v, v]
            k = r-c*np.outer(x, x)
            jmax = 9
            acols = [np.linalg.matrix_power(r, i)@x for i in range(jmax)]
            ss = [x@a for a in acols]
            gamma = [acols[0]]
            for t in range(1, jmax):
                gamma.append(acols[t]-c*sum(ss[i]*gamma[t-1-i] for i in range(t)))
                assert np.max(np.abs(gamma[t]-np.linalg.matrix_power(k, t)@x)) < 2e-10
                assert gamma[t].min() >= -2e-10
            drops = []
            for j in range(1, jmax+1):
                decrement = c*sum(acols[i]*gamma[j-1-i] for i in range(j))
                target = np.diag(np.linalg.matrix_power(r, j)-np.linalg.matrix_power(k, j))
                err = np.max(np.abs(decrement-target))
                worst = max(worst, err)
                assert err < 2e-10 and decrement.min() >= -2e-10
                drops.append(target)
                cases += 1
            # Build genuinely inexact selected columns and enclose them by residuals.
            ahat = [a+rng.normal(size=len(r))*1e-8 for a in acols]
            lower_eig = np.linalg.eigvalsh(xmat)[0]
            norms = []
            previous = np.eye(len(r))[:, v]
            old_error = 0.0
            for i in range(jmax):
                residual = previous-xmat@ahat[i]
                error = (old_error+np.linalg.norm(residual))/lower_eig
                assert np.linalg.norm(ahat[i]-acols[i]) <= error+1e-13
                norms.append(error)
                previous, old_error = ahat[i], error
            ai = [np.array([a-e, a+e]) for a, e in zip(ahat, norms)]
            ci = np.array([1/(ahat[0][v]+norms[0]), 1/(ahat[0][v]-norms[0])])[:, None]
            si = []
            for i in range(jmax):
                center = ahat[0]@ahat[i]
                radius = np.linalg.norm(ahat[0])*norms[i]+np.linalg.norm(ahat[i])*norms[0]+norms[0]*norms[i]
                si.append(np.array([center-radius, center+radius])[:, None])
            intervals = decrement_intervals(ai, ci, si, jmax)
            for true, interval in zip(drops, intervals):
                assert np.all(true >= interval[0]-2e-10)
                assert np.all(true <= interval[1]+2e-10)
            # Signed cache errors remain fixed for exact decrements, even if large relatively.
            signed = rng.uniform(-.01, .01, (jmax, len(r)))
            initial = np.array([np.diag(np.linalg.matrix_power(r, j)) for j in range(1, jmax+1)])
            updated_cache = initial+signed-np.array(drops)
            true_updated = np.array([np.diag(np.linalg.matrix_power(k, j)) for j in range(1, jmax+1)])
            assert np.max(np.abs(updated_cache-true_updated-signed)) < 2e-10
    # Mirror inc's monotone flags and count: selections only use refreshed candidates.
    n, selected, seen, total = 30, set(), set(), 0
    for _ in range(8):
        fresh = [u for u in range(n) if u not in seen]
        batch = list(rng.choice(fresh, min(3, len(fresh)), replace=False))
        assert not (seen & set(batch))
        total += len(batch)
        seen.update(batch)
        selected.add(min(seen-selected))
        assert total == len(seen)
        assert len(set(range(n))-seen) == n-total
        assert not ((set(range(n))-seen) & selected)
    # Read-only night-run observations; each record is n, distinct refresh count, computed never count.
    # runs/{com-amazon,com-dblp}.ungraph.inc1_{s0,ac}.json; runs/s1b/com-amazon.ungraph.inc2.json.
    records = [(334863, 736, 334126), (334863, 1232, 333630),
               (317080, 256, 316823), (317080, 448, 316631),
               (334863, 1504, 333358)]
    percentages = []
    for n, refreshed, never in records:
        assert n-1-refreshed == never
        percentages.append(100*never/(n-1))
    print(f'PASS T11 incremental: {cases} moment decrements through j=9; max error={worst:.3e} < 2e-10; '
          'residual/interval and flag checks pass; never-refreshed percentages='+
          ', '.join(f'{x:.3f}' for x in percentages))


run('T11 independent incremental decrements', test)
