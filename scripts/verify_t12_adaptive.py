"""T12 uniform JL event, including signs and all deterministic grounding sets."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))
from math import comb, ceil, log


def test():
    rng = np.random.default_rng(1212)
    count = 0
    max_error = 0.0
    # Check the explicit Rademacher mgf/fourth-moment inequalities by full sign enumeration.
    for n in [2, 4, 6]:
        signs = np.array(list(itertools.product([-1., 1.], repeat=n)))
        for _ in range(5):
            w = rng.normal(size=n); w /= np.linalg.norm(w)
            z = signs@w
            assert abs(np.mean(z*z)-1) < 2e-12
            assert np.mean(z**4) <= 3+2e-12
            for t in [.01, .1, .25]:
                assert np.mean(np.exp(t*z*z)) <= (1-2*t)**(-.5)+2e-12
                assert np.mean(np.exp(-t*z*z)) <= np.exp(-t+1.5*t*t)+2e-12
    for a in fixtures()[:-1]:
        n, k, p = len(a), 2, 4
        jmax, depth = p+1, (p+2)//2
        zeta, delta = .3, .01
        total = sum(comb(n, t)*(n-t) for t in range(k+1))
        d = ceil(8/zeta**2*log(2*jmax*total/delta))
        fresh = ceil(8/zeta**2*log(2*jmax*n*k/delta))
        assert 2*jmax*total*np.exp(-d*zeta*zeta/8) <= delta*(1+1e-12)
        assert 2*jmax*n*k*np.exp(-fresh*zeta*zeta/8) <= delta*(1+1e-12)
        assert total <= n*(k+1)*n**k and d >= fresh
        cc = np.linalg.cholesky(a).T
        for dist in ['gaussian', 'sign']:
            draw = lambda: (rng.normal(size=(d, n)) if dist == 'gaussian'
                            else rng.choice([-1., 1.], size=(d, n)))/np.sqrt(d)
            g, h = draw(), draw()
            seen = 0
            for t in range(k+1):
                for selected in itertools.combinations(range(n), t):
                    u = [i for i in range(n) if i not in selected]
                    r = np.zeros_like(a); r[np.ix_(u, u)] = np.linalg.inv(a[np.ix_(u, u)])
                    for j in range(1, jmax+1):
                        ell = (j+1)//2
                        factor = g@cc if j%2 else h
                        y = factor@np.linalg.matrix_power(r, ell)
                        truth = np.diag(np.linalg.matrix_power(r, j))
                        err = np.max(np.abs(np.sum(y[:, u]**2, axis=0)/truth[u]-1))
                        max_error = max(max_error, err)
                        assert err < zeta
                        count += len(u)
                    seen += len(u)
            assert seen == total
    # Marginal precision and epsilon specializations across a broad parameter grid.
    for p in [1, 2, 4, 8, 32, 64]:
        for eta in [.01, .1, .5, .9]:
            zeta = eta/(16*p)
            assert np.expm1(p*np.log1p(4*zeta/(1-zeta))) <= eta
    for eps in [.001, .01, .1, .5, .9]:
        zeta, theta = eps/8, eps/2
        chi = (1+zeta)/(1-zeta)
        alpha = min(chi**-2, (1-theta)/chi)
        assert 1-np.exp(-alpha) >= 1-np.exp(-1)-eps
    print(f'PASS T12 adaptive JL: {count} all-set moment/candidate checks for Gaussian and sign rows; '
          f'max relative error={max_error:.4f} < 0.3; explicit dimensions, mgf bounds and precision pass')


run('T12 adaptive uniform moment accuracy', test)
