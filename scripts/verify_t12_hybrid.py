"""T12 hybrid exact retention and capped/acceptance stopping, including noisy scoring."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def inverse(a, selected):
    u = [i for i in range(len(a)) if i not in selected]
    r = np.zeros_like(a); r[np.ix_(u, u)] = np.linalg.inv(a[np.ix_(u, u)])
    return r


def delta(r, v, p):
    k = r-np.outer(r[:, v], r[:, v])/r[v, v]
    return np.trace(np.linalg.matrix_power(r, p))-np.trace(np.linalg.matrix_power(k, p))


def marginal(moments, p):
    # p[z^p] log(1 + sum r_{j+1}/r_1 z^j), using the T3 scalar recurrence.
    aa = [1.]+[moments[j]/moments[0] for j in range(1, p+1)]
    ll = [0.]*(p+1)
    for j in range(1, p+1):
        ll[j] = aa[j]-sum(i*ll[i]*aa[j-i] for i in range(1, j))/j
    return p*ll[p]


def pick_round(keys, true, known, budget, theta, rb=2):
    remaining = sorted(keys, key=lambda u: (-keys[u], u))
    first = remaining[0]
    processed, newly = [], []
    best = None
    stop = ''
    while remaining:
        while remaining and remaining[0] in known:
            u = remaining.pop(0); processed.append(u)
            if best is None or true[u] > true[best]: best = u
        top = keys[remaining[0]] if remaining else -1
        if best is not None and true[best] >= (1-theta)*top:
            stop = 'accept'; break
        if best is not None and budget and len(newly) >= budget:
            stop = 'cap'; break
        batch = []
        while remaining and len(batch) < rb:
            u = remaining.pop(0); processed.append(u)
            if u not in known: batch.append(u); newly.append(u)
            elif best is None or true[u] > true[best]: best = u
        for u in batch:
            known.add(u)
            if best is None or true[u] > true[best]: best = u
    if not stop: stop = 'empty'
    assert best is not None
    if stop == 'cap': assert first in processed
    return best, newly, stop


def test():
    rng = np.random.default_rng(1213)
    rounds = runs = caps = accepts = 0
    mingain = 1.
    for a in fixtures()[:-1]:
        n, k = len(a), 3
        c0 = np.linalg.cholesky(a).T
        for p in [1, 2, 4]:
            d = 4096
            g = rng.choice([-1., 1.], size=(d, n))/np.sqrt(d)
            h = rng.choice([-1., 1.], size=(d, n))/np.sqrt(d)
            depth = (p+2)//2
            for budget, theta in [(0, 0.), (1, 0.), (2, .1), (1, .3)]:
                known, selected, refreshed = set(), set(), 0
                scalar = {}
                r = inverse(a, selected)
                base = np.trace(np.linalg.matrix_power(r, p))
                opt = base-min(np.trace(np.linalg.matrix_power(inverse(a, ss), p))
                               for ss in itertools.combinations(range(n), k))
                eta = .45
                alpha = min((1-eta)/(1+eta), (1-theta)*(1-eta))
                for step in range(k):
                    active = [u for u in range(n) if u not in selected]
                    ys = [[f@np.linalg.matrix_power(r, ell) for ell in range(1, depth+1)]
                          for f in [g@c0, h]]
                    estimates = []
                    for j in range(1, p+2):
                        y = ys[0 if j%2 else 1][(j-1)//2]
                        estimates.append(np.sum(y*y, axis=0))
                    true = {u: delta(r, u, p) for u in active}
                    keys = {u: scalar[u] if u in known else marginal([m[u] for m in estimates], p) for u in active}
                    assert all(abs(keys[u]/true[u]-1) <= eta for u in active)
                    v, new, stop = pick_round(keys, true, known, budget, theta)
                    caps += stop == 'cap'; accepts += stop == 'accept'
                    refreshed += len(new)
                    assert refreshed == len(known) <= n
                    assert true[v] >= alpha*max(true.values())-2e-8
                    # Independent exact decrement update of all retained entries.
                    x = r[:, v].copy(); cc = 1/x[v]
                    newr = r-cc*np.outer(x, x)
                    for u in known-selected-{v}:
                        mm = []
                        for j in range(1, p+2):
                            decrement = cc*sum((np.linalg.matrix_power(r, i)@x)[u]
                                                *(np.linalg.matrix_power(newr, j-1-i)@x)[u] for i in range(j))
                            mm.append(np.linalg.matrix_power(r, j)[u, u]-decrement)
                        scalar[u] = marginal(mm, p)
                        assert abs(scalar[u]-delta(newr, u, p)) <= 2e-8*max(1, abs(scalar[u]))
                    selected.add(v); r = inverse(a, selected)
                    rounds += 1
                gain = base-np.trace(np.linalg.matrix_power(r, p))
                mingain = min(mingain, gain/opt)
                assert gain >= (1-np.exp(-alpha))*opt-2e-8
                runs += 1
    # A flat landscape forces the budget branch even though every key is accurate.
    control_keys = {0: 1.2, 1: 1.15, 2: 1.1}
    chosen, new, stop = pick_round(control_keys, {0: 1., 1: 1., 2: 1.}, set(), 1, 0., rb=1)
    assert stop == 'cap' and chosen == 0 and new == [0]
    control_caps = 1
    # Adversarial score arrays check the sufficient inexact-score loss, independent of sketches.
    for _ in range(1000):
        n = 11; eta, nu, theta = .2, .07, .1
        truth = np.exp(rng.uniform(-2, 2, n))
        exact_scores = truth*(1+rng.uniform(-nu, nu, n))
        known = {i for i in range(n) if rng.random() < .4}
        keys = {i: exact_scores[i] if i in known else truth[i]*(1+rng.uniform(-eta, eta)) for i in range(n)}
        picked, _, stop = pick_round(keys, dict(enumerate(exact_scores)), known, 1, theta)
        control_caps += stop == 'cap'
        ee = max(eta, nu)
        alpha = min((1-ee)/(1+ee), (1-theta)*(1-ee))*(1-nu)/(1+nu)
        assert truth[picked] >= alpha*max(truth)-2e-12
    assert control_caps > 0 and accepts > 0
    print(f'PASS T12 hybrid: {runs} runs/{rounds} adaptive rounds, {caps} cap stops/{accepts} acceptance stops; '
          f'min gain/OPT={mingain:.6f}; exact-retention error < 2e-8; '
          f'1001 score controls ({control_caps} cap stops) pass')


run('T12 hybrid capped greedy', test)
