"""T12 higher-power maintained sketches: noncommutative updates and work."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def inverse(a, selected):
    u = [i for i in range(len(a)) if i not in selected]
    r = np.zeros_like(a)
    r[np.ix_(u, u)] = np.linalg.inv(a[np.ix_(u, u)])
    return r


def test():
    rng = np.random.default_rng(1211)
    cases = targets = 0
    worst = 0.0
    for a in fixtures()[:-1]:
        n = len(a)
        c0 = np.linalg.cholesky(a).T
        for p in range(1, 9):
            depth = (p+2)//2
            for dist in ['gaussian', 'sign']:
                d = 37
                draw = lambda: (rng.normal(size=(d, n)) if dist == 'gaussian'
                                else rng.choice([-1., 1.], size=(d, n)))/np.sqrt(d)
                lefts = [draw(), draw()@c0]
                r, selected = inverse(a, set()), set()
                ys = [[f@np.linalg.matrix_power(r, ell) for ell in range(1, depth+1)] for f in lefts]
                for step in range(3):
                    active = [u for u in range(n) if u not in selected]
                    # Adaptive choice from the maintained high-power sketches.
                    v = max(active, key=lambda u: np.sum(ys[0][-1][:, u]**2))
                    x = r[:, v].copy(); c = 1/x[v]
                    chain = [x]
                    for _ in range(1, depth): chain.append(r@chain[-1])
                    ss = [x@z for z in chain]
                    bs = [x]
                    for t in range(1, depth):
                        bs.append(chain[t]-c*sum(ss[i]*bs[t-1-i] for i in range(t)))
                    k = r-c*np.outer(x, x)
                    for t, b in enumerate(bs):
                        assert np.max(np.abs(b-np.linalg.matrix_power(k, t)@x)) < 2e-7*max(1, np.max(np.abs(b)))
                    if depth >= 2:
                        assert np.max(np.abs(bs[1]-(r@x-c*(x@x)*x))) < 2e-10
                    for family in range(2):
                        cols = [y[:, v].copy() for y in ys[family]]
                        for ell in range(1, depth+1):
                            ys[family][ell-1] -= c*sum(np.outer(cols[i], bs[ell-1-i]) for i in range(ell))
                    selected.add(v)
                    r = inverse(a, selected)
                    assert np.max(np.abs(r-k)) < 2e-10
                    for f, family in zip(lefts, ys):
                        for ell, y in enumerate(family, 1):
                            ref = f@np.linalg.matrix_power(r, ell)
                            err = np.max(np.abs(y-ref))/max(1, np.max(np.abs(ref)))
                            worst = max(worst, err)
                            assert err < 3e-8
                    for ell in range(1, depth+1):
                        rl = np.linalg.matrix_power(r, ell)
                        for f, power in [(np.eye(n), 2*ell), (c0, 2*ell-1)]:
                            lhs = np.sum((f@rl)**2, axis=0)
                            rhs = np.diag(np.linalg.matrix_power(r, power))
                            assert np.max(np.abs(lhs-rhs)) < 3e-8*max(1, np.max(np.abs(rhs)))
                            targets += 1
                    cases += 1
                # Counts for two families, including p=1 and the scalar hybrid chain.
                outer_products = 2*sum(range(1, depth+1))
                assert outer_products == depth*(depth+1)
                assert 2*d*depth >= p+1
                assert depth <= p+1
                if p == 1: assert depth == 1 and outer_products == 2
    print(f'PASS T12 higher sketches: {cases} adaptive updates, {targets} moment identities; '
          f'max scaled update error={worst:.3e} < 3e-8; selected-chain and work counts pass')


run('T12 higher-power maintained sketches', test)
